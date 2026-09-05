"""hyp.manip dispatcher: resolve a manipulator spec (or chain of specs) and
fit_transform (or, for an already-fitted spec, transform) it.

Wrapped by datawrangler's funnel so any input (array / DataFrame / list / text
/ polars) arrives as DataFrame(s); the resolved Manipulator (sklearn-compatible,
DataFrame-based) is applied directly rather than via the array-based
core.apply_model.

Model specs may be: a registered name (``MANIPULATORS``' ``__name__``\\ s, or
-- inside a `list` spec only -- any name in the combined manip/reduce/align/
cluster registry, GH #153), a dict in either
``{'model': ..., 'params': {...}}`` (LEGACY) or the canonical
``{'model': ..., 'args': [...], 'kwargs': {...}}`` form, a Manipulator
subclass, a Manipulator instance, a `list` of any of these (chained via
`hypertools.Pipeline`, GH #274/#153), or an already-fitted Manipulator/
`Pipeline` (routed to `.transform` instead of `.fit_transform` -- the
no-re-fitting path behind ``return_model=True``).
"""
import datawrangler as dw
import numpy as np
import pandas as pd

from .common import Manipulator
from .normalize import Normalize
from .zscore import ZScore
from .smooth import Smooth
from .resample import Resample
from .delay import Delay
from ..core.shared import unpack_model, require_data, no_observations_message
from ..core.pipeline import Pipeline


MANIPULATORS = [Normalize, ZScore, Smooth, Resample, Delay]


def _supported_names():
    return [m.__name__ for m in MANIPULATORS]


def _unknown_model_message(name):
    # recommend the CANONICAL dict form, not the deprecated {'model':
    # ..., 'params': {...}} one (audit F14-017)
    return (f"unknown manip model {name!r}; supported names: "
            f"{', '.join(_supported_names())} (or pass a dict "
            "{'model': ..., 'args': [...], 'kwargs': {...}}, a Manipulator "
            "subclass, or a Manipulator instance directly)")


def _validate_manip_input(data):
    """Catch unsupported/degenerate inputs BEFORE the datawrangler funnel
    mangles them into nonsense (audit F14-005/F14-007/X2-error-quality-004).

    - `None` raised a bare ``IndexError: list index out of range`` from deep
      inside datawrangler; it now gets the same clear `TypeError` the rest of
      the API raises.
    - a pandas `Series` was wrangled to an EMPTY (0, 0) DataFrame, producing
      actively misleading downstream errors; it is coerced to a
      single-column DataFrame instead (matching `hyp.normalize`).
    - empty inputs (empty DataFrame/array, empty list) raised bare
      ``IndexError``\\ s -- or, for ``[]``, loaded a topic-model text corpus
      before failing inside sklearn's LDA; they now get a clear `ValueError`.
    """
    no_observations = no_observations_message('manipulate')
    require_data(data, 'manip')
    if isinstance(data, tuple):
        # a tuple of datasets is accepted exactly like a list (final wave
        # item 15: it used to leak a raw IndexError from the funnel)
        data = list(data)
    if isinstance(data, pd.Series):
        return data.to_frame()
    if isinstance(data, (pd.DataFrame, np.ndarray)) and data.shape[0] == 0:
        raise ValueError(no_observations)
    if isinstance(data, list):
        if len(data) == 0:
            raise ValueError(no_observations)
        data = [d.to_frame() if isinstance(d, pd.Series) else d for d in data]
        for d in data:
            if isinstance(d, (pd.DataFrame, np.ndarray)) and d.shape[0] == 0:
                raise ValueError(no_observations)
    return data


def _resolve_single_step(model, kwargs):
    """Resolve one (non-list) manip spec into a fit/transform-capable
    Manipulator instance (or an already-fitted instance/other duck-typed
    object passed straight through)."""
    resolved = unpack_model(model, valid=MANIPULATORS, parent_class=Manipulator)

    if isinstance(resolved, type):
        return resolved(**kwargs)

    if isinstance(resolved, dict):
        cls = resolved["model"]
        args = resolved.get("args", [])
        step_kwargs = resolved.get("kwargs", {})
        if isinstance(cls, str):
            # an unresolved name inside a dict spec (audit F14-010): give the
            # same clear error as the bare-string path below instead of
            # leaking "'str' object has no attribute 'fit_transform'"
            raise ValueError(_unknown_model_message(cls))
        if isinstance(cls, type):
            return cls(*args, **step_kwargs)
        if step_kwargs and hasattr(cls, "set_params"):
            cls.set_params(**step_kwargs)
        return cls

    if isinstance(resolved, str):
        raise ValueError(_unknown_model_message(resolved))

    # an already-constructed (or already-fitted) instance
    return resolved


@dw.decorate.funnel
def _funneled_manip(data, model="ZScore", return_model=False, normalize=None,
                    reduce=None, ndims=None, align=None, cluster=None,
                    **kwargs):
    """Funnel-decorated core of `manip` (see `manip`'s docstring); `manip`
    validates raw input first, then delegates here so datawrangler's funnel
    only ever sees inputs it handles sensibly."""
    # cross-module stage kwargs (#138): manip is the FIRST stage in the
    # canonical order (manip -> normalize -> reduce -> align -> cluster), so a
    # manip call carrying any downstream stage kwarg assembles + runs a Pipeline
    # via build_pipeline -- parity with normalize()/analyze() (QC 2026-07: these
    # kwargs used to fall through **kwargs into the manipulator's constructor
    # and raise TypeError).
    if any(s is not None for s in (normalize, reduce, align, cluster)):
        from ..core.pipeline import build_pipeline
        # fold the manipulator's own constructor kwargs into a dict spec so they
        # still reach the manip stage (model=None -- an explicitly skipped
        # manip stage -- is passed through so build_pipeline omits the stage)
        manip_spec = ({'model': model, 'kwargs': dict(kwargs)}
                      if (kwargs and model is not None
                          and not isinstance(model, (list, Pipeline)))
                      else model)
        pipe = build_pipeline(manip=manip_spec, normalize=normalize,
                              reduce=reduce, ndims=ndims, align=align,
                              cluster=cluster)
        result = pipe.fit_transform(data)
        return (result, pipe) if return_model else result

    if isinstance(model, list):
        pipeline = Pipeline(model)
        result = pipeline.fit_transform(data)
        return (result, pipeline) if return_model else result

    if isinstance(model, Pipeline):
        result = model.transform(data) if model.is_fitted else model.fit_transform(data)
        return (result, model) if return_model else result

    resolved = _resolve_single_step(model, kwargs)

    if isinstance(resolved, Manipulator) and resolved.is_fitted:
        result = resolved.transform(data)
    else:
        result = resolved.fit_transform(data)

    return (result, resolved) if return_model else result


def manip(data, model="ZScore", return_model=False, normalize=None, reduce=None,
          ndims=None, align=None, cluster=None, **kwargs):
    """Apply a manipulation (or chain of manipulations) to `data`.

    Manipulations run in native (per-dataset) space -- no dataset is ever
    mixed row-wise with another: `Smooth` and `Resample` are applied
    independently to each dataset in a list. `ZScore` and `Normalize`
    transform each dataset separately too, but fit ONE shared set of
    statistics (mean/std, or baseline/peak) across every dataset in a
    list -- like ``hypertools.normalize``'s ``'across'`` mode. There is
    currently no manip-level option for within-dataset statistics on a
    list; call `manip` on each dataset separately for that.

    Parameters
    ----------
    data : DataFrame/array or list/tuple of these
        Dataset(s) to manipulate. A pandas `Series` is treated as a
        single-column dataset; a tuple of datasets is treated exactly like
        a list. `None` raises a `TypeError`.

    model : str, dict, class, instance, list, Pipeline, False, or None
        Which manipulator(s) to apply (default: `'ZScore'`). `False` or
        `None` skips the manipulation entirely and returns the input
        unchanged (matching `align`/`cluster`/`reduce`'s skip contract).

        - A string is one of `MANIPULATORS`' names (Normalize, ZScore,
          Smooth, Resample, Delay).
        - A dict may be the canonical
          ``{'model': ..., 'args': [...], 'kwargs': {...}}`` or the LEGACY
          ``{'model': ..., 'params': {...}}`` form (accepted for backward
          compatibility, but emits a `DeprecationWarning`).
        - A bare (uninstantiated) Manipulator subclass, or an
          already-constructed (unfitted) instance, is used directly.
        - A `list` chains its elements into a `hypertools.Pipeline`
          (GH #274/#153) and runs `fit_transform` end to end -- e.g.
          ``model=[{'model': 'Smooth', 'kwargs': {'kernel_width': 25}},
          {'model': 'Resample', 'kwargs': {'n_samples': 1000}}, 'ZScore']``.
          Inside a list, each string is resolved first against
          `MANIPULATORS`, then the reduce, align, and cluster registries
          (in that order, GH #153) -- so
          ``model=['Smooth', 'UMAP']`` and ``model=['Smooth', 'HyperAlign']``
          both work, via `hypertools.core.pipeline.Pipeline`.
        - An ALREADY-FITTED Manipulator instance or `Pipeline` (returned
          from a previous `manip(..., return_model=True)` call) is applied
          to `data` via `.transform` (its learned parameters, e.g.
          `ZScore`'s fitted mean/std, are reused -- not re-estimated).

    return_model : bool
        If True, also return the fitted (or reused) model: the fitted
        `Manipulator` when `model` was a single spec, or a fitted
        `hypertools.Pipeline` when `model` was a list (default: False).

    normalize, reduce, align, cluster : model spec, False, or None
        Cross-module stage kwargs (GH #138): when any of these is given,
        the other stages also run (via
        `hypertools.core.pipeline.build_pipeline`), in the canonical order
        `manip -> normalize -> reduce -> align -> cluster` (GH #153), with
        this function's own `model=` slotted in at the manip stage
        (default: None for all four, i.e. only `manip` runs). `False`
        skips a stage, exactly like None.

    ndims : int or None
        Passed through to the `reduce` stage (as `ndims=`) when `reduce=`
        is also given.

    **kwargs
        Passed through to the manipulator's constructor when `model`
        resolves to a class (ignored when `model` is a list, an already
        -instantiated instance, or a fitted model/Pipeline being reused).

    Returns
    -------
    The manipulated data (and the fitted model/Pipeline if
    `return_model=True`).

    Notes
    -----
    `manip` and `hypertools.normalize` follow different conventions for
    the same input: `manip` returns DataFrames (index/columns preserved)
    while `normalize` returns numpy arrays; `manip` propagates NaNs while
    `normalize` PPCA-imputes them at format time; `manip` z-scores with
    the sample std (``ddof=1``) while `normalize` uses the population std
    (``ddof=0``); and a 1-D array is treated as a single ROW by `manip`'s
    data funnel but as a single COLUMN by `normalize`.

    Examples
    --------
    >>> import numpy as np
    >>> import hypertools as hyp
    >>> x = np.arange(20, dtype=float).reshape(10, 2)
    >>> z = hyp.manip(x, model='ZScore')
    >>> np.allclose(z.mean(axis=0), 0.0)
    True
    >>> smoothed = hyp.manip(x, model='Smooth', kernel_width=5)
    >>> smoothed.shape
    (10, 2)
    >>> chained = hyp.manip(x, model=[{'model': 'Resample',
    ...                                'kwargs': {'n_samples': 50}},
    ...                               'ZScore'])
    >>> chained.shape
    (50, 2)
    """
    data = _validate_manip_input(data)

    # False is an explicit "skip this stage", for the model spec and every
    # stage kwarg, exactly like None -- the same contract align/cluster/
    # reduce follow (2026-07 release audit, final wave item 7: manip(x,
    # model=False) used to raise "unknown model spec: got a bool instance").
    if model is False:
        model = None
    normalize = None if normalize is False else normalize
    reduce = None if reduce is False else reduce
    align = None if align is False else align
    cluster = None if cluster is False else cluster

    if model is None and all(s is None for s in (normalize, reduce, align,
                                                 cluster)):
        # nothing to do: hand the (validated) input back unchanged,
        # matching reduce(reduce=None)/cluster(cluster=None)/align(
        # model=None)
        return (data, None) if return_model else data

    import warnings
    with warnings.catch_warnings():
        # TARGETED upstream-noise filter (release-1.0 audit,
        # X1-api-consistency-018 / X4-warnings-002): pydata-wrangler's
        # stacked-apply/funnel path passes the deprecated `copy=` keyword
        # to `pd.concat` (datawrangler/decorate/decorate.py `pandas_stack`),
        # so every MULTI-DATASET manip emitted a pandas
        # Pandas4Warning/DeprecationWarning the user can do nothing about.
        # Suppress ONLY that specific message around our stacked-apply call
        # site -- never Pandas4Warning wholesale -- until the fix lands
        # upstream in pydata-wrangler (same maintainers).
        warnings.filterwarnings(
            'ignore', message='The copy keyword is deprecated',
            category=DeprecationWarning)
        return _funneled_manip(data, model=model, return_model=return_model,
                               normalize=normalize, reduce=reduce, ndims=ndims,
                               align=align, cluster=cluster, **kwargs)
