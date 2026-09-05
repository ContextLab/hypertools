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
from ..core.model import external_stacklevel


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


def _warn_deprecated_alias(name):
    """Emit the promised DeprecationWarning for the documented-deprecated
    `'hyper'` alias (release-1.0 audit, X1-api-consistency-020: the
    docstring called it deprecated but no warning ever fired, so users
    could never learn the canonical name). The other aliases ('SRM',
    'DetSRM', 'RSRM') are documented as plain -- non-deprecated --
    aliases and stay silent."""
    if name == 'hyper':
        warnings.warn(
            "model='hyper' is a deprecated alias for 'HyperAlign'; pass "
            "model='HyperAlign' instead.",
            DeprecationWarning, stacklevel=external_stacklevel())


def _reject_unknown_aligner(name):
    """Raise a clear error for an unknown align-model name. `unpack_model`
    passes unrecognized strings through unchanged, so without this the string
    later hit "'str' object has no attribute 'fit_transform'" (QC 2026-07).
    Matches the clear "unknown X model" errors reduce/cluster/predict give."""
    raise ValueError(
        f"unknown align model {name!r}; supported names: "
        f"{', '.join(sorted(a.__name__ for a in ALIGNERS))} (or pass an "
        "Aligner subclass or instance directly).")


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
        `None` when `model` is `None` or `False` (i.e. skip alignment
        entirely -- docstrings across the toolbox promise `align=False`
        as a no-op, matching `normalize=False`).
    """
    if model is None or model is False:
        return None
    if model is True:
        raise ValueError(
            "model=True is not a valid align spec; pass the algorithm name "
            "instead (e.g. model='HyperAlign'), or False/None to skip "
            "alignment.")

    if isinstance(model, str):
        _warn_deprecated_alias(model)
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
        if c_model is None or c_model is False:
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
                DeprecationWarning, stacklevel=external_stacklevel())
            c_args, c_kwargs = [], dict(model['params'])
        else:
            c_args, c_kwargs = [], {}
        if isinstance(c_model, str):
            _warn_deprecated_alias(c_model)
            c_model = _ALIAS.get(c_model, c_model)
        resolved_inner = unpack_model(c_model, valid=ALIGNERS, parent_class=Aligner)
        if isinstance(resolved_inner, str):
            # unknown name inside the DICT form -- previously slipped past the
            # bare-string guard below and hit the cryptic AttributeError
            # (QC 2026-07 red-team: align={'model': 'Nope'}).
            _reject_unknown_aligner(resolved_inner)
        if isinstance(resolved_inner, type):
            return resolved_inner(*c_args, **c_kwargs)
        # already-constructed (or already-fitted) instance: params ignored,
        # used as-is
        return resolved_inner

    resolved = unpack_model(model, valid=ALIGNERS, parent_class=Aligner)
    if isinstance(resolved, str):
        _reject_unknown_aligner(resolved)
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


def _compute_score(return_score, score_metric, before_data, after_data):
    """Build the `{'before', 'after', 'metric'}` score dict for `align`'s
    `return_score=True` (GH #285), or `None` when `return_score` is False.

    `before_data`/`after_data` are each either a single DataFrame/array or a
    list of them (whatever `align` had on hand at that return point); both
    are normalized to list form before delegating to
    `hypertools.align.score.alignment_score`, which raises a clear
    `ValueError` for ragged (unequal-shape) input.
    """
    if not return_score:
        return None
    from .score import alignment_score
    before_list = before_data if isinstance(before_data, list) else [before_data]
    after_list = after_data if isinstance(after_data, list) else [after_data]
    return alignment_score(before_list, aligned=after_list, metric=score_metric)


def _build_return(result, return_model, model, return_score, score):
    """Assemble `align`'s return value from whichever of `return_model=`/
    `return_score=` are set (GH #285).

    Order (documented on `align`'s docstring): `result` alone; `(result,
    model)` for `return_model=True` only (unchanged from before GH #285, so
    existing callers are unaffected); `(result, score)` for `return_score=True`
    only; `(result, model, score)` when both are True.
    """
    if return_model and return_score:
        return result, model, score
    if return_model:
        return result, model
    if return_score:
        return result, score
    return result


@dw.decorate.funnel
def _align(data, model='HyperAlign', return_model=False,
           return_score=False, score_metric='dispersion',
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

    # the funnel coerces a single (non-list) input to a single DataFrame and
    # keeps lists as lists; remember which so the output can match the
    # input's list/single-item shape (like hyp.reduce -- the Aligner always
    # builds a list internally, which used to up-promote single inputs to a
    # list-of-one, F12-align-004)
    was_list = isinstance(data, list)

    # a whole already-fitted Pipeline handed back as model= (e.g. the model
    # from an earlier cross-module return_model=True call) is reused as-is via
    # .transform, BEFORE _resolve_align_spec below -- otherwise unpack_model
    # raises "unknown model: Pipeline" (QC 2026-07). Redundant stage kwargs are
    # warned + ignored (the Pipeline already encodes them).
    # the alignment-quality score (GH #285) compares equal-shape data before
    # vs. after the ALIGN stage specifically; that pairing is undefined
    # inside a multi-stage Pipeline (manip/normalize/reduce also reshape and
    # rescale the data), so return_score= is rejected up front rather than
    # silently scoring something else.
    if return_score and any(
            stage is not None for stage in (manip, normalize, reduce, cluster)):
        raise ValueError(
            "return_score=True is only supported for the plain align stage; "
            "it cannot be combined with manip=/normalize=/reduce=/cluster= "
            "(the alignment-quality score compares data before vs. after "
            "alignment specifically, which is undefined inside a multi-stage "
            "pipeline). Call hyp.align(data, model=..., return_score=True) "
            "on its own.")

    from ..core.shared import is_reused_pipeline
    if is_reused_pipeline(model, {'manip': manip, 'normalize': normalize,
                                  'reduce': reduce, 'cluster': cluster}, 'model'):
        raw = _to_arrays(model.transform(data))
        result = _match_input_shape(raw, was_list)
        score = _compute_score(return_score, score_metric, data, raw)
        return _build_return(result, return_model, model, return_score, score)

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
        result = _match_input_shape(pipeline.fit_transform(data), was_list)
        return (result, pipeline) if return_model else result

    if resolved is None:
        # no-op (model=None or model=False): hand the data back unchanged
        # (numerically identical, converted to the array/list shape align
        # always returns) -- no trimming, padding, or format_data pass
        raw = _to_arrays(data)
        result = _match_input_shape(raw, was_list)
        score = _compute_score(return_score, score_metric, data, raw)
        return _build_return(result, return_model, None, return_score, score)

    if format_data:
        data = _apply_format_data(data)

    # an already-fitted Aligner (returned from an earlier
    # return_model=True call) is reused via `transform`, never refit
    if isinstance(resolved, Aligner) and resolved.is_fitted:
        raw = _to_arrays(resolved.transform(data))
        result = _match_input_shape(raw, was_list)
        score = _compute_score(return_score, score_metric, data, raw)
        return _build_return(result, return_model, resolved, return_score, score)

    raw = _to_arrays(resolved.fit_transform(data))
    result = _match_input_shape(raw, was_list)
    score = _compute_score(return_score, score_metric, data, raw)
    return _build_return(result, return_model, resolved, return_score, score)


def _match_input_shape(result, was_list):
    """Unwrap a list-of-one `result` when the ORIGINAL input was a single
    (non-list) dataset, so `align(bare_array)` returns a bare array like
    `hyp.reduce`/`hyp.cluster` and align's own Returns docstring promise
    (F12-align-004). List inputs (of any length) are returned untouched."""
    if not was_list and isinstance(result, list) and len(result) == 1:
        return result[0]
    return result


def _to_arrays(result):
    """Convert an `Aligner`'s DataFrame(s) output to numpy array(s) -- the
    classic `hyp.align` API has always returned arrays (see
    `hypertools.tools.align.align`), and every OTHER dispatcher (`reduce`,
    `cluster`, `manip`) returns arrays too."""
    if isinstance(result, list):
        return [np.asarray(r) for r in result]
    return np.asarray(result)


def align(data, model='HyperAlign', return_model=False,
          return_score=False, score_metric='dispersion',
          manip=None, normalize=None, reduce=None, ndims=None, cluster=None,
          format_data=True, **kwargs):
    """
    Aligns a list of datasets into a shared coordinate space.

    Resolves `model` (and any cross-module stage kwargs) into a fitted
    `Aligner` and applies it, following the same model-spec grammar as
    `hypertools.reduce.reduce.reduce`/`hypertools.cluster.cluster.cluster`.

    Parameters
    ----------
    data : numpy array, pandas/polars DataFrame, or list/tuple of these
        The datasets to align. Any input format is funneled into
        DataFrame(s) before dispatch (a tuple of datasets is treated
        exactly like a list). Rows are matched across datasets by index
        value and returned in the FIRST dataset's index order; datasets
        with DUPLICATED index labels keep only the first row per label
        (with a `UserWarning`), so output rows always match one-to-one
        across datasets. `None` raises a `TypeError`; an empty list raises
        a `ValueError`.

    model : str, class, instance, dict, fitted Aligner, False, or None
        Alignment algorithm to use. Supported names: `'HyperAlign'`
        (hyperalignment, Haxby et al. 2011; `'hyper'` is a deprecated
        alias and emits a `DeprecationWarning` naming the canonical
        spelling), `'SharedResponseModel'` (`'SRM'` alias),
        `'DeterministicSharedResponseModel'` (`'DetSRM'` alias),
        `'RobustSharedResponseModel'` (`'RSRM'` alias), `'Procrustes'`, and
        `'NullAlign'` (returns the trimmed/padded data unchanged). Can be
        passed as a string, a bare (uninstantiated) `Aligner` subclass, an
        already-constructed instance (the classes are importable as e.g.
        `from hypertools.align import HyperAlign, Procrustes,
        SharedResponseModel, NullAlign`), the canonical dict spec
        `{'model': ..., 'args': [...], 'kwargs': {...}}`, or the LEGACY
        dict spec `{'model': ..., 'params': {...}}` (accepted for backward
        compatibility, but emits a `DeprecationWarning`). A
        previously-fitted `Aligner` (as returned by `return_model=True`) is
        applied via `.transform` instead of being refit. `False` or `None`
        skips alignment entirely and returns the data unchanged
        (`model=True` raises a `ValueError` -- name an algorithm instead).
        (default: `'HyperAlign'`).

    return_model : bool
        If True, also return the fitted model: the fitted `Aligner` when
        only the `align` stage ran, or a fitted `hypertools.Pipeline` when
        `manip=`/`normalize=`/`reduce=`/`cluster=` made multiple stages run
        (default: False).

    return_score : bool
        If True, also return an alignment-quality score dict comparing the
        data before vs. after alignment (GH #285): see
        `hypertools.align.score.alignment_score` for the two supported
        `score_metric=` values (`'dispersion'`, the default, and `'isc'`).
        Only supported for the plain align stage -- raises `ValueError` if
        combined with `manip=`/`normalize=`/`reduce=`/`cluster=`, since the
        before/after pairing is undefined inside a multi-stage pipeline
        (default: False).

    score_metric : {'dispersion', 'isc'}
        Which score `return_score=True` computes -- see
        `hypertools.align.score.alignment_score` (default: `'dispersion'`).

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
        `'HyperAlign'`, `features=` for the SRM family). Keyword arguments
        the model does not accept raise a `TypeError` naming them (they
        used to be silently ignored, so a typo'd parameter went unnoticed).
        `align=` is also accepted here as a DEPRECATED alias for `model=`
        (emits a `DeprecationWarning`; passing both `model=` -- with a
        non-default value -- and `align=` raises `ValueError`), preserving
        the classic `hyp.plot(..., align='hyper')`-style call.

    Returns
    -------
    aligned : list of numpy arrays (or a single array)
        The aligned data, in the same list/single-item shape as `data`
        (matching `hyp.reduce`/`hyp.cluster` and the classic `hyp.align`
        API, which return numpy arrays; note that `hyp.manip` instead
        returns pandas DataFrames): a list input
        returns a list, a single bare array/DataFrame returns a single
        array. Output rows follow the FIRST dataset's index order. The
        return value depends on `return_model=`/`return_score=`: neither ->
        `aligned`; `return_model=True` only -> `(aligned, model)`;
        `return_score=True` only -> `(aligned, score)`; both -> `(aligned,
        model, score)` (`model` always comes right after `aligned`, matching
        the pre-existing `return_model=True` return shape, with `score`
        appended last).

    Raises
    ------
    ValueError
        If `data` is an empty list (nothing to align), if any dataset has
        more than 2 dimensions (pass a list of 2-D observations-by-
        features datasets, not a 3-D stack), if the datasets
        share no common row-index values, or if `model` is an unknown
        name / `True`.
    TypeError
        If `data` is `None`, or a keyword argument is not accepted by the
        resolved model's constructor (e.g. a misspelled parameter name).

    Examples
    --------
    >>> import numpy as np
    >>> import hypertools as hyp
    >>> rng = np.random.default_rng(0)
    >>> a = rng.standard_normal((30, 4))
    >>> b = a @ np.linalg.qr(rng.standard_normal((4, 4)))[0]  # rotated copy
    >>> aligned = hyp.align([a, b], model='HyperAlign')
    >>> [d.shape for d in aligned]
    [(30, 4), (30, 4)]

    """
    from ..core.shared import require_data, no_observations_message
    # None always raises the unified dispatcher TypeError (it used to leak a
    # misleading "input has no observations (0 rows)" from format_data --
    # 2026-07 release audit, final wave item 9)
    require_data(data, 'align')
    if isinstance(data, tuple):
        # a tuple of datasets is accepted exactly like a list (final wave
        # item 15: it used to be funneled as one opaque object and die with
        # the same misleading no-observations error)
        data = list(data)
    if isinstance(data, list) and len(data) == 0:
        # an empty list used to fall through to the TEXT input funnel,
        # downloading the minipedia corpus and dying with a cryptic
        # LatentDirichletAllocation error (QC 2026-07, F12-align-006)
        raise ValueError(
            no_observations_message('align', 'got an empty list')
            + ' Pass a list of one or more numeric arrays/DataFrames to '
            'align.')
    if isinstance(data, list) and type(data) is not list:
        # GH #209: normalize a list SUBCLASS to a plain `list` before the
        # funnel-decorated `_align` runs -- see `_align`'s docstring.
        data = list(data)
    # reject 3-D (or higher) array input up front (release-1.0 audit,
    # X3-performance-004): datawrangler's funnel silently FLATTENED a
    # (slices x time x channels) stack into (slices, time*channels), so a
    # natural mistake -- passing a 3-D array instead of a list of 2-D
    # datasets -- ground through a huge dense alignment and returned
    # meaningless output. Matches reduce/cluster's fail-fast shape checks.
    _datasets = data if isinstance(data, list) else [data]
    for _i, _d in enumerate(_datasets):
        _ndim = getattr(_d, 'ndim', None)
        if _ndim is None and not isinstance(_d, (str, bytes)):
            try:
                _ndim = np.ndim(_d)
            except Exception:
                _ndim = None
        if _ndim is not None and _ndim > 2:
            _which = f'dataset {_i}' if isinstance(data, list) else 'data'
            raise ValueError(
                f'align expects 2-D (observations x features) datasets, '
                f'but {_which} has {_ndim} dimensions '
                f'(shape {tuple(np.shape(_d))}). Pass a LIST of 2-D '
                'arrays/DataFrames (e.g. one per subject) instead of a '
                'higher-dimensional stack -- e.g. list(x) for a 3-D '
                'array x.')
    return _align(data, model=model, return_model=return_model,
                  return_score=return_score, score_metric=score_metric,
                  manip=manip, normalize=normalize, reduce=reduce,
                  ndims=ndims, cluster=cluster, format_data=format_data,
                  **kwargs)
