"""hyp.impute dispatcher: resolve an imputer spec and fit_transform it.

Mirrors `hypertools.manip.manip`'s dispatcher shape (and, for `return_model`,
`hypertools.predict.predict`'s convention) but for same-shape missing-data
imputation. The public `impute` first normalizes obviously-degenerate or
ambiguous inputs (None, scalars, empty inputs, 1-D series, infinite values
-- see `impute`'s data docs), then hands off to a datawrangler-funnel-
wrapped core so any input (array / DataFrame / list / text / polars)
arrives as DataFrame(s); the resolved Imputer (DataFrame-based) is applied
directly.

Model specs may be: a registered name (`IMPUTERS`' `__name__`\\ s), a dict in
either `{'model': ..., 'params': {...}}` (deprecated) or the fork-style
`{'model': ..., 'args': [...], 'kwargs': {...}}` form, an Imputer subclass,
or an Imputer instance. An instance that has already been fit
(`instance.is_fitted`) is routed to `Imputer.transform` instead of
`fit_transform` -- the no-re-fitting path behind `return_model=True`: a
fitted imputer returned by an earlier `impute(..., return_model=True)` call
can be passed back as `model=` on NEW data without re-estimating its learned
parameters.
"""
import numbers
import warnings

import numpy as np
import pandas as pd
import datawrangler as dw

from .common import Imputer
from .ppca import PPCA
from .sklearn_imputers import SimpleImputer, KNNImputer, IterativeImputer
from .kalman import Kalman
from ..core.shared import supported_names, unpack_model


IMPUTERS = [PPCA, SimpleImputer, KNNImputer, IterativeImputer, Kalman]


def _spec_help():
    return (f'supported names: {", ".join(supported_names(IMPUTERS))} (or pass a '
            "dict {'model': ..., 'args': [...], 'kwargs': {...}}, an Imputer "
            'subclass, or an Imputer instance directly). Note that raw '
            'scikit-learn imputer classes/instances are not accepted; use '
            'the hypertools wrappers of the same name (e.g. '
            'hypertools.impute.SimpleImputer).')


def _coerce_dataset(d):
    """Normalize ONE dataset-like object before wrangling: a 1-D array or a
    pandas Series is a UNIVARIATE series -- n observations of 1 feature,
    i.e. an (n, 1) column -- matching format_data/plot's convention. (QC
    2026-07 red-team F17-impute-010: the funnel used to wrangle a (4,)
    array into ONE row of 4 features, whose NaN then became an all-missing
    "column" silently filled with 0.0 instead of the series statistic.)"""
    if isinstance(d, pd.Series):
        return d.to_frame()
    if isinstance(d, np.ndarray) and d.ndim == 1:
        return d.reshape(-1, 1)
    return d


def _check_finite_observed(d):
    """Reject infinite values with format_data's clear message instead of a
    raw sklearn 'Input X contains infinity' (or PPCA/Kalman silently
    treating inf as data) -- QC 2026-07 red-team F17-impute-015."""
    try:
        values = np.asarray(d, dtype=float)
    except (TypeError, ValueError):
        return  # non-numeric (e.g. text) data: let the funnel handle it
    if np.isinf(values).any():
        raise ValueError(
            'input contains infinite values; remove or replace them '
            '(e.g. with np.nan for missing entries) before imputing.')


def _normalize_data(data):
    """Validate/normalize `data` before it reaches the datawrangler funnel,
    turning degenerate inputs into clear errors instead of cryptic library
    internals (QC 2026-07 red-team F17-impute-008/-010,
    X2-error-quality-004)."""
    from ..core.shared import require_data, no_observations_message
    require_data(data, 'impute')
    if isinstance(data, tuple):
        # a tuple of datasets is accepted exactly like a list (2026-07
        # release audit, final wave item 15: it used to be funneled as one
        # opaque object and die with a misleading all-NaN error)
        data = list(data)
    if isinstance(data, (bool, np.bool_)) or isinstance(data, numbers.Number):
        raise ValueError(
            f'cannot impute a single scalar observation ({data!r}); pass a '
            'dataset with at least 1 row and 1 column.')
    if isinstance(data, np.ndarray):
        if data.ndim == 0:
            raise ValueError(
                f'cannot impute a single scalar observation ({data!r}); pass '
                'a dataset with at least 1 row and 1 column.')
        if data.size == 0:
            raise ValueError(
                f'input has no observations (got an array of shape '
                f'{tuple(data.shape)}); there is nothing to impute.')
        data = _coerce_dataset(data)
        _check_finite_observed(data)
        return data
    if isinstance(data, pd.DataFrame) and (data.shape[0] == 0 or data.shape[1] == 0):
        raise ValueError(
            f'input has no observations (got a DataFrame of shape '
            f'{tuple(data.shape)}); there is nothing to impute.')
    if isinstance(data, list):
        if len(data) == 0:
            raise ValueError(
                'input has no observations (got an empty list); there is '
                'nothing to impute.')
        # a flat list of numbers is a single univariate series (see
        # `_coerce_dataset`); NaN is a float, so it satisfies Number.
        if all(isinstance(v, numbers.Number) and not isinstance(v, (bool, np.bool_))
               for v in data):
            data = np.asarray(data, dtype=float).reshape(-1, 1)
            _check_finite_observed(data)
            return data
        data = [_coerce_dataset(d) for d in data]
        for i, d in enumerate(data):
            # an EMPTY dataset inside a list used to slip through and get
            # the misleading "entirely missing (all values are NaN)" error
            # -- or silently poison a mixed list (2026-07 release audit,
            # final wave item 13). Empty and all-NaN are now distinguished.
            shape = getattr(d, 'shape', None)
            if shape is not None and len(shape) >= 1 and (
                    shape[0] == 0 or (len(shape) > 1 and shape[1] == 0)):
                raise ValueError(no_observations_message(
                    'impute', f'dataset {i} has shape {tuple(shape)}'))
            _check_finite_observed(d)
        return data
    data = _coerce_dataset(data)
    _check_finite_observed(data)
    return data


def _all_missing(data):
    """Whether every numeric value across `data` (a wrangled DataFrame or
    list of them) is NaN. Empty datasets do NOT count as all-missing
    (``isnan(empty).all()`` is vacuously True, which used to mislabel a
    0-row input as "entirely missing" -- 2026-07 release audit, final wave
    item 13)."""
    datasets = data if isinstance(data, list) else [data]
    try:
        values = [np.asarray(d, dtype=float) for d in datasets]
    except (TypeError, ValueError):
        return False
    return all(v.size > 0 and np.isnan(v).all() for v in values)


def _mismatched_columns(data):
    """Whether `data` is a list of (wrangled) datasets that do NOT share
    columns -- joint (stacked) imputation is impossible for those."""
    if not isinstance(data, list) or len(data) < 2:
        return False
    if not all(isinstance(d, pd.DataFrame) for d in data):
        return False
    first = list(data[0].columns)
    return any(list(d.columns) != first for d in data[1:])


@dw.decorate.funnel
def _wrangled_impute(data, model='PPCA', return_model=False, **kwargs):
    """Funnel-wrapped core of `impute` (see its docstring)."""
    if isinstance(data, list) and len(data) == 0:
        raise ValueError('input has no observations (got an empty list); '
                         'there is nothing to impute.')
    if _all_missing(data):
        raise ValueError(
            'input is entirely missing (all values are NaN); there is '
            'nothing to impute from.')

    if _mismatched_columns(data):
        # pd.concat over mismatched columns silently WIDENED every dataset
        # to the column union and "filled" the invented all-NaN blocks with
        # the other datasets' statistics (QC 2026-07 red-team
        # F17-impute-001). Joint imputation is impossible without shared
        # columns, so impute each dataset independently instead.
        if isinstance(model, Imputer):
            raise ValueError(
                'datasets in a list must share columns to be imputed '
                f'jointly; got columns {[list(d.columns) for d in data]}. '
                'An Imputer instance can only be fit once, so impute each '
                'dataset in its own impute() call (or pass the model as a '
                'name/class/dict spec to fit one imputer per dataset).')
        warnings.warn(
            'datasets do not share columns, so they cannot be imputed '
            'jointly; imputing each dataset independently instead. (Shared '
            'columns are required to pool information across datasets.)')
        results = [_wrangled_impute(d, model=model, return_model=return_model,
                                    **kwargs) for d in data]
        if return_model:
            return [r[0] for r in results], [r[1] for r in results]
        return results

    # a column with NO observed values at all carries no information: the
    # imputers can only fabricate values for it (Kalman/PPCA fill 0.0).
    # Warn so a dead sensor's zeros are not mistaken for data (release-1.0
    # audit, D09-tutorials-applied-012). Checked on the POOLED view -- for
    # a list sharing columns, a column observed in ANY dataset is informed
    # (the datasets are stacked and imputed jointly).
    _datasets = data if isinstance(data, list) else [data]
    if all(isinstance(d, pd.DataFrame) for d in _datasets):
        try:
            _stacked = pd.concat(_datasets, axis=0)
            _vals = _stacked.to_numpy(dtype=float)
        except (TypeError, ValueError):
            _vals = None
        if _vals is not None and _vals.size:
            _dead = np.isnan(_vals).all(axis=0)
            if _dead.any():
                _names = [str(c) for c, d_ in zip(_stacked.columns, _dead)
                          if d_]
                warnings.warn(
                    f'column(s) {_names} have no observed values at all; '
                    'their "imputed" values are not informed by any data '
                    "(Kalman and PPCA fill such columns with 0.0). Drop "
                    'these columns, or treat their filled values as '
                    'placeholders rather than data.', UserWarning)

    if isinstance(model, dict) and 'kwargs' not in model and 'args' not in model:
        # {'model': ..., 'params': {...}} form: unpack before handing the
        # inner model spec to unpack_model (which only auto-unpacks the
        # fork-style {'model', 'args', 'kwargs'} triple). Warn here (round17
        # Task 6 fix): this shortcut used to bypass unpack_model entirely
        # for this dict shape, silently swallowing its DeprecationWarning.
        if 'model' not in model:
            raise ValueError(
                f"a model spec dict must include a 'model' key; got keys "
                f'{sorted(model)}. Pass e.g. '
                "{'model': 'PPCA', 'kwargs': {...}}.")
        if 'params' in model:
            warnings.warn(
                "{'model': ..., 'params': {...}} is deprecated; use "
                "{'model': ..., 'args': [...], 'kwargs': {...}} instead",
                DeprecationWarning, stacklevel=2)
        kwargs = {**dict(model.get('params', {})), **kwargs}
        model = model['model']
    elif isinstance(model, dict) and 'model' not in model:
        raise ValueError(
            f"a model spec dict must include a 'model' key; got keys "
            f'{sorted(model)}. Pass e.g. '
            "{'model': 'PPCA', 'kwargs': {...}}.")

    if isinstance(model, str) and model not in supported_names(IMPUTERS):
        # case-insensitive fallback (release-1.0 audit,
        # D09-tutorials-applied-014: model='ppca' used to raise "unknown
        # impute model 'ppca'" while listing 'PPCA' as supported)
        _by_lower = {n.lower(): n for n in supported_names(IMPUTERS)}
        model = _by_lower.get(model.lower(), model)
    try:
        resolved = unpack_model(model, valid=IMPUTERS, parent_class=Imputer)
    except ValueError as e:
        # unpack_model's terse "unknown model: 42" (for non-string specs)
        # gave no hint about accepted spec forms (QC 2026-07 red-team
        # F17-impute-011); re-raise with the same rich message unknown
        # strings get.
        raise ValueError(f'unknown impute model {model!r}; {_spec_help()}') from e

    if isinstance(resolved, type):
        resolved = resolved(**kwargs)
    elif isinstance(resolved, dict):
        cls = resolved['model']
        # merge outer **kwargs into the dict form's kwargs (outer kwargs win,
        # matching the deprecated params-dict form); they used to be silently
        # dropped for this spec form only (QC 2026-07 red-team
        # F17-impute-007).
        resolved = cls(*resolved.get('args', []),
                       **{**resolved.get('kwargs', {}), **kwargs})
    elif isinstance(resolved, str):
        raise ValueError(f'unknown impute model {resolved!r}; {_spec_help()}')
    elif kwargs:
        # an already-constructed instance cannot absorb constructor kwargs;
        # warn instead of silently ignoring them (QC 2026-07 red-team).
        warnings.warn(
            f'ignoring keyword argument(s) {sorted(kwargs)}: model= is '
            'already a constructed instance, so constructor parameters '
            'cannot be applied. Pass the class (or a name/dict spec) to '
            'set parameters.')

    if isinstance(resolved, Imputer) and resolved.is_fitted:
        result = resolved.transform(data)
    else:
        result = resolved.fit_transform(data)

    return (result, resolved) if return_model else result


def impute(data, model='PPCA', return_model=False, **kwargs):
    """Fill missing (NaN) values in `data`, preserving its shape.

    Observed (non-NaN) values are preserved exactly by every model; only
    the missing positions are filled. Output values are float64 (even on a
    NaN-free passthrough).

    Parameters
    ----------
    data : DataFrame/array or list/tuple of these
        Dataset(s) to impute. A 1-D array, a flat list of numbers, or a
        pandas Series is treated as a UNIVARIATE series -- n observations
        of 1 feature, i.e. an (n, 1) column -- matching
        `hyp.plot`/`format_data`'s convention; a tuple of datasets is
        treated exactly like a list. A list of datasets SHARING
        COLUMNS is stacked (row-wise) and imputed jointly, then split back
        into a list matching the input structure (each dataset keeps its
        own index); datasets that do NOT share columns cannot be pooled
        and are imputed independently instead (with a warning).
        Degenerate inputs (None, a scalar, empty or entirely-NaN data,
        infinite values) raise a clear error; empty and all-NaN datasets
        are distinguished (an empty dataset -- alone or inside a list --
        raises the no-observations error naming its shape, not the
        all-NaN one).

    model : str, dict, class, or Imputer instance
        Which imputer to use (default: 'PPCA', matching the pre-1.0
        `format_data` default). A string is one of `IMPUTERS`' names (PPCA,
        SimpleImputer, KNNImputer, IterativeImputer, Kalman); names are
        matched case-insensitively ('ppca' works too). A dict may be
        `{'model': ..., 'params': {...}}` (deprecated) or
        `{'model': ..., 'args': [...], 'kwargs': {...}}`. A class or an
        already-constructed (unfitted) instance is used directly. An
        ALREADY-FITTED Imputer instance (returned from a previous
        `return_model=True` call) is applied to `data` via `transform`
        (its learned parameters are reused -- not re-estimated).

    return_model : bool
        If True, also return the fitted (or reused) Imputer instance, so it
        can be passed back as `model=` on future calls with new data
        (default: False). (When mismatched-column datasets are imputed
        independently, a LIST of fitted imputers is returned instead.)

    **kwargs
        Passed through to the imputer's constructor when `model` resolves
        to a name, class, or dict spec (merged into a dict spec's own
        ``kwargs``, with these outer values winning). When `model` is
        already a constructed instance they cannot be applied, and a
        warning is raised. Unknown parameter names raise `TypeError`
        (typos used to be swallowed silently).

    Returns
    -------
    The imputed data (and the fitted Imputer if return_model=True). Lists
    in, lists out: a single input dataset returns a single imputed
    DataFrame.

    Examples
    --------
    >>> import numpy as np
    >>> import hypertools as hyp
    >>> x = np.cumsum(np.random.default_rng(0).standard_normal((40, 5)),
    ...               axis=0)
    >>> x[3, 1] = np.nan
    >>> filled = hyp.impute(x, model='PPCA')
    >>> filled.shape
    (40, 5)
    >>> bool(np.isnan(filled.to_numpy()).any())
    False
    """
    data = _normalize_data(data)
    return _wrangled_impute(data, model=model, return_model=return_model,
                            **kwargs)
