"""hyp.align dispatcher: resolve an aligner spec and fit_transform it.

Wrapped by datawrangler's funnel so any input (array / DataFrame / list / text /
polars) arrives as DataFrame(s); the resolved Aligner (list-based, sklearn-
compatible) is applied directly. NOT routed through core.apply_model, whose
stack-and-fit-once recipe is wrong for aligning a *list* to a shared template.
"""
import warnings

import numpy as np
import pandas as pd
import datawrangler as dw

from .common import Aligner
from .hyperalign import HyperAlign
from .procrustes import Procrustes
from .srm import SharedResponseModel, DeterministicSharedResponseModel, RobustSharedResponseModel
from .null import NullAlign
from ..core.shared import unpack_model


ALIGNERS = [HyperAlign, SharedResponseModel, DeterministicSharedResponseModel,
            RobustSharedResponseModel, Procrustes, NullAlign]

#: legacy (pre-1.0 / dev-1.0) string spellings -> canonical registry names.
#: `hypertools.tools.align.align` (the classic API shim) used to translate
#: these itself before calling this dispatcher; now the dispatcher accepts
#: them directly (as `model=`, or via the deprecated `align=` kwarg alias
#: below) so both entry points share one translation table.
_ALIAS = {
    'hyper': 'HyperAlign',
    'HyperAlign': 'HyperAlign',
    'SRM': 'SharedResponseModel',
    'SharedResponseModel': 'SharedResponseModel',
    'DetSRM': 'DeterministicSharedResponseModel',
    'DeterministicSharedResponseModel': 'DeterministicSharedResponseModel',
    'RSRM': 'RobustSharedResponseModel',
    'RobustSharedResponseModel': 'RobustSharedResponseModel',
    'Procrustes': 'Procrustes',
    'NullAlign': 'NullAlign',
}


def _resolve_align_spec(model, extra_kwargs):
    """Resolve a `model=` spec (+ leftover `**kwargs`) into an unfitted (or,
    for an already-fitted/-constructed instance, passed-through-unchanged)
    `Aligner`.

    Mirrors `hypertools.reduce.reduce`'s inline dict-spec handling (rather
    than delegating dict specs to `core.shared.unpack_model`, whose
    canonical-dict branch requires ALL of `'model'`/`'args'`/`'kwargs'` to
    be present) so `{'model': ..., 'kwargs': {...}}` (no `'args'`) -- as
    well as the canonical `{'model': ..., 'args': [...], 'kwargs': {...}}`
    and the LEGACY `{'model': ..., 'params': {...}}` (accepted with a
    `DeprecationWarning`) -- all work.

    Parameters
    ----------
    model : str, class, instance, dict, or None
        The `model=` spec (see `align`'s docstring for the full grammar).
    extra_kwargs : dict
        Leftover `**kwargs` from the `align()` call, forwarded to the
        resolved class's constructor -- but only when `model` is a bare
        registry name/class (a dict spec's own `'args'`/`'kwargs'` take
        precedence over these, and an already-constructed instance ignores
        them entirely, matching `hyp.reduce`/`hyp.cluster`).

    Returns
    -------
    Aligner or None
        `None` when `model` is `None` (i.e. skip alignment entirely).
    """
    if model is None:
        return None

    if isinstance(model, str):
        model = _ALIAS.get(model, model)

    if isinstance(model, dict):
        try:
            c_model = model['model']
        except KeyError:
            raise ValueError(
                "If passing a dictionary, pass the model as the value of "
                "the 'model' key and a dictionary of custom params as the "
                "value of the 'kwargs' key (the legacy 'params' key is "
                "also accepted)."
            )
        if c_model is None:
            return None
        if 'args' in model or 'kwargs' in model:
            c_args = list(model.get('args', []))
            c_kwargs = dict(model.get('kwargs', {}))
        elif 'params' in model:
            # LEGACY form (dev-1.0/fork): accepted for backward
            # compatibility, but deprecated in favor of the canonical
            # {'model', 'args', 'kwargs'} triple above.
            warnings.warn(
                "{'model': ..., 'params': {...}} is deprecated; use "
                "{'model': ..., 'args': [...], 'kwargs': {...}} instead",
                DeprecationWarning, stacklevel=3)
            c_args, c_kwargs = [], dict(model['params'])
        else:
            c_args, c_kwargs = [], {}
        if isinstance(c_model, str):
            c_model = _ALIAS.get(c_model, c_model)
        resolved_inner = unpack_model(c_model, valid=ALIGNERS, parent_class=Aligner)
        if isinstance(resolved_inner, type):
            return resolved_inner(*c_args, **c_kwargs)
        # already-constructed (or already-fitted) instance: params ignored,
        # used as-is
        return resolved_inner

    resolved = unpack_model(model, valid=ALIGNERS, parent_class=Aligner)
    if isinstance(resolved, type):
        return resolved(**extra_kwargs)
    # an already-constructed (unfitted) or already-fitted instance is
    # passed through unchanged (the caller checks `.is_fitted` to decide
    # whether to fit_transform or reuse via transform)
    return resolved


def _apply_format_data(data):
    """Run the missing-data-fill / text-to-matrix pass
    (`hypertools.tools.format_data.format_data`) on already-DataFrame'd
    `data` (from the `@dw.decorate.funnel` wrapper below), then re-wrap the
    resulting arrays as DataFrames -- preserving each original dataset's row
    index -- since `Aligner.fit`/`transform` require DataFrame(s) for
    `datawrangler.unstack`/`trim_and_pad`.
    """
    from ..tools.format_data import format_data as formatter

    was_list = isinstance(data, list)
    items = data if was_list else [data]
    formatted = formatter(items, ppca=True)
    rewrapped = [
        pd.DataFrame(np.asarray(arr), index=getattr(orig, 'index', None))
        for arr, orig in zip(formatted, items)
    ]
    return rewrapped if was_list else rewrapped[0]


@dw.decorate.funnel
def _align(data, model='HyperAlign', return_model=False,
           manip=None, normalize=None, reduce=None, ndims=None, cluster=None,
           format_data=True, **kwargs):
    """`@dw.decorate.funnel`-wrapped implementation of `align` (below).

    Split out so the public `align()` can normalize list-SUBCLASS input to
    a plain `list` (GH #209: `datawrangler.wrangle`'s own `type(x) is
    list` check -- rather than `isinstance` -- silently mistreats a list
    subclass as a single opaque object) BEFORE the funnel decorator's
    `wrangle()` call runs, since a decorator cannot be bypassed once
    invoked. See `align`'s docstring for the full parameter/return
    documentation.
    """
    # deprecated align= alias for model= (kept for hyp.plot(align=...) and
    # any other pre-1.0-style caller that never migrated to model=). Since
    # `model` keeps its literal 'HyperAlign' default (required by the
    # signature), "both given" is detected as align= together with a
    # NON-default model= -- passing align= alongside the (unchanged)
    # default is treated as the caller only using the deprecated kwarg.
    if 'align' in kwargs:
        legacy_model = kwargs.pop('align')
        if model != 'HyperAlign':
            raise ValueError(
                "cannot pass both model= and the deprecated align= kwarg; "
                "use model= only (align= is a deprecated alias for it)."
            )
        warnings.warn(
            "align= is deprecated as a model-spec kwarg name on "
            "hypertools.align.align.align; use model= instead (e.g. "
            "hyp.align(data, model='hyper')).",
            DeprecationWarning, stacklevel=2,
        )
        model = legacy_model

    resolved = _resolve_align_spec(model, kwargs)

    # cross-module kwargs (#138): assemble and run a Pipeline (in canonical
    # order, #153) instead of the single-stage path below whenever another
    # stage is requested. Lazy import avoids an align<->core.pipeline cycle
    # (core.pipeline itself lazily imports align.align).
    if any(stage is not None for stage in (manip, normalize, reduce, cluster)):
        from ..core.pipeline import build_pipeline
        pipeline = build_pipeline(manip=manip, normalize=normalize,
                                   reduce=reduce, ndims=ndims,
                                   align=resolved, cluster=cluster)
        result = pipeline.fit_transform(data)
        return (result, pipeline) if return_model else result

    if resolved is None:
        return (data, None) if return_model else data

    if format_data:
        data = _apply_format_data(data)

    # an already-fitted Aligner (returned from an earlier
    # return_model=True call) is reused via `transform`, never refit
    if isinstance(resolved, Aligner) and resolved.is_fitted:
        result = _to_arrays(resolved.transform(data))
        return (result, resolved) if return_model else result

    result = _to_arrays(resolved.fit_transform(data))
    return (result, resolved) if return_model else result


def _to_arrays(result):
    """Convert an `Aligner`'s DataFrame(s) output to numpy array(s) -- the
    classic `hyp.align` API has always returned arrays (see
    `hypertools.tools.align.align`), and every OTHER dispatcher (`reduce`,
    `cluster`, `manip`) returns arrays too."""
    if isinstance(result, list):
        return [np.asarray(r) for r in result]
    return np.asarray(result)


def align(data, model='HyperAlign', return_model=False,
          manip=None, normalize=None, reduce=None, ndims=None, cluster=None,
          format_data=True, **kwargs):
    """
    Aligns a list of datasets into a shared coordinate space.

    Resolves `model` (and any cross-module stage kwargs) into a fitted
    `Aligner` and applies it, following the same model-spec grammar as
    `hypertools.reduce.reduce.reduce`/`hypertools.cluster.cluster.cluster`.

    Parameters
    ----------
    data : numpy array, pandas/polars DataFrame, or list of these
        The datasets to align. Any input format is funneled into
        DataFrame(s) before dispatch.

    model : str, class, instance, dict, or fitted Aligner
        Alignment algorithm to use. Supported names: `'HyperAlign'`
        (hyperalignment, Haxby et al. 2011; `'hyper'` is a deprecated
        alias), `'SharedResponseModel'` (`'SRM'` alias),
        `'DeterministicSharedResponseModel'` (`'DetSRM'` alias),
        `'RobustSharedResponseModel'` (`'RSRM'` alias), `'Procrustes'`, and
        `'NullAlign'` (returns the trimmed/padded data unchanged). Can be
        passed as a string, a bare (uninstantiated) `Aligner` subclass, an
        already-constructed instance, the canonical dict spec
        `{'model': ..., 'args': [...], 'kwargs': {...}}`, or the LEGACY
        dict spec `{'model': ..., 'params': {...}}` (accepted for backward
        compatibility, but emits a `DeprecationWarning`). A
        previously-fitted `Aligner` (as returned by `return_model=True`) is
        applied via `.transform` instead of being refit (default:
        `'HyperAlign'`).

    return_model : bool
        If True, also return the fitted model: the fitted `Aligner` when
        only the `align` stage ran, or a fitted `hypertools.Pipeline` when
        `manip=`/`normalize=`/`reduce=`/`cluster=` made multiple stages run
        (default: False).

    manip, normalize, reduce, cluster : model spec or None
        Cross-module stage kwargs (GH #138): when any of these is given,
        the other stages also run (via
        `hypertools.core.pipeline.build_pipeline`), in the canonical order
        `manip -> normalize -> reduce -> align -> cluster` (GH #153), with
        this function's own `model=`/`n_iter=`/etc. slotted in at the align
        stage (default: None for all four, i.e. only `align` runs).

    ndims : int or None
        Passed through to the `reduce` stage (as `ndims=`) when `reduce=`
        is also given.

    format_data : bool
        Whether or not to first run the missing-data-fill / text-to-matrix
        `format_data` pass (default: True).

    **kwargs
        Extra keyword arguments forwarded to `model`'s constructor when
        `model` is a bare registry name/class (e.g. `n_iter=` for
        `'HyperAlign'`, `features=` for the SRM family). `align=` is also
        accepted here as a DEPRECATED alias for `model=` (emits a
        `DeprecationWarning`; passing both `model=` -- with a non-default
        value -- and `align=` raises `ValueError`), preserving the classic
        `hyp.plot(..., align='hyper')`-style call.

    Returns
    -------
    aligned : list of numpy arrays (or a single array)
        The aligned data, in the same list/single-item shape as `data`
        (matching `hyp.reduce`/`hyp.cluster`/`hyp.manip`, and the classic
        `hyp.align` API, which have always returned arrays). If
        `return_model=True`, an `(aligned, model)` tuple is returned
        instead.

    """
    if isinstance(data, list) and type(data) is not list:
        # GH #209: normalize a list SUBCLASS to a plain `list` before the
        # funnel-decorated `_align` runs -- see `_align`'s docstring.
        data = list(data)
    return _align(data, model=model, return_model=return_model,
                  manip=manip, normalize=normalize, reduce=reduce,
                  ndims=ndims, cluster=cluster, format_data=format_data,
                  **kwargs)
