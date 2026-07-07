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

from .common import Manipulator
from .normalize import Normalize
from .zscore import ZScore
from .smooth import Smooth
from .resample import Resample
from ..core.shared import unpack_model
from ..core.pipeline import Pipeline
from ..core.configurator import apply_defaults


MANIPULATORS = [Normalize, ZScore, Smooth, Resample]


def _supported_names():
    return [m.__name__ for m in MANIPULATORS]


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
        if isinstance(cls, type):
            return cls(*args, **step_kwargs)
        if step_kwargs and hasattr(cls, "set_params"):
            cls.set_params(**step_kwargs)
        return cls

    if isinstance(resolved, str):
        raise ValueError(
            f"unknown manip model {resolved!r}; supported names: "
            f"{', '.join(_supported_names())} (or pass a dict "
            "{'model': ..., 'params': {...}}, a Manipulator subclass, or a "
            "Manipulator instance directly)")

    # an already-constructed (or already-fitted) instance
    return resolved


@dw.decorate.funnel
def manip(data, model="ZScore", return_model=False, **kwargs):
    """Apply a per-dataset manipulation (or chain of manipulations) to `data`.

    Parameters
    ----------
    data : DataFrame/array or list of these
        Dataset(s) to manipulate.

    model : str, dict, class, Manipulator instance, list, or Pipeline
        Which manipulator(s) to apply (default: `'ZScore'`).

        - A string is one of `MANIPULATORS`' names (Normalize, ZScore,
          Smooth, Resample).
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

    **kwargs
        Passed through to the manipulator's constructor when `model`
        resolves to a class (ignored when `model` is a list, an already
        -instantiated instance, or a fitted model/Pipeline being reused).

    Returns
    -------
    The manipulated data (and the fitted model/Pipeline if
    `return_model=True`).
    """
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
