#!/usr/bin/env python
"""Unified stack/unstack model application -- the hypertools 1.0 core.

Every model-applying operation in hypertools follows the same recipe:

    format -> STACK the list of datasets into one array -> apply the model
    ONCE -> UNSTACK the result back into the input's list structure

Fitting a single model to the stacked data is what makes results comparable
across datasets (separate per-dataset fits would produce incomparable
embeddings/labels). `apply_model` exposes that recipe directly, supporting
model specification as a registry name, a dict with params, a scikit-learn
style instance, or a list of any of these (applied as a pipeline).

The registry is an explicit whitelist -- the earlier refactor resolved model
strings with eval() over sklearn's namespace, which was fragile and
security-sensitive; named models here are imported explicitly.
"""

import inspect
import os
import sys

import numpy as np


#: the hypertools package directory, used by `external_stacklevel` to tell
#: library-internal frames apart from user code
_PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def external_stacklevel(extra_internal=('datawrangler',)):
    """Compute a `warnings.warn` stacklevel that points at USER code.

    Walks the call stack outward from the caller of this function and
    returns the `stacklevel` of the first frame OUTSIDE the hypertools
    package (and outside the `extra_internal` packages -- datawrangler's
    funnel wrappers sit between hypertools dispatchers on several call
    paths). Use as ``warnings.warn(..., stacklevel=external_stacklevel())``.

    Why: warnings raised while resolving specs threaded through
    ``hyp.plot()``/``hyp.analyze()`` used to be attributed (via a fixed
    stacklevel) to hypertools' own frames, and Python's default filters
    only DISPLAY a `DeprecationWarning` when it is attributed to
    ``__main__`` -- so e.g. the legacy ``{'model': ..., 'params': {...}}``
    deprecation was invisible to `plot()` callers (release-1.0 audit,
    X4-warnings-008/-009).
    """
    try:
        frame = sys._getframe(1)
    except ValueError:  # pragma: no cover - no caller frame at all
        return 2
    level = 1
    while frame is not None:
        filename = frame.f_code.co_filename
        internal = filename.startswith(_PACKAGE_DIR + os.sep)
        if not internal:
            internal = any(os.sep + pkg + os.sep in filename
                           for pkg in extra_internal)
        if not internal:
            return level
        frame = frame.f_back
        level += 1
    return max(level, 2)


def _build_registry():
    from ..reduce.common import REDUCERS
    from ..cluster.common import CLUSTERERS
    registry = {}
    # REDUCERS already includes the mixture/soft-clustering models (GH
    # #174) shared with hyp.cluster; CLUSTERERS adds the hard-clustering
    # models on top.
    registry.update(REDUCERS)
    registry.update(CLUSTERERS)
    # UMAP resolves lazily (importing umap triggers slow numba JIT)
    registry['UMAP'] = None
    return registry


def supported_models():
    """Names accepted as string model specifications."""
    return sorted(_build_registry().keys())


def apply_model(data, model, mode='auto', return_model=False,
                format_data=True, stack=True, ndims=None):
    """Apply a model (or pipeline of models) to one or more datasets.

    Parameters
    ----------
    data : numpy array, pandas DataFrame, or list of arrays/DataFrames
        The dataset(s) to transform.

    model : str, class, dict, sklearn-style instance, or list
        - str: a registered model name (see
          `hypertools.core.supported_models()`), e.g. 'PCA', 'KMeans',
          'GaussianMixture'. (This registry covers the REDUCE and CLUSTER
          model families only; manipulator, aligner, forecaster, and
          imputer names -- e.g. 'ZScore', 'HyperAlign', 'Kalman' -- are
          handled by their own dispatchers, `hyp.manip`/`hyp.align`/
          `hyp.predict`/`hyp.impute`, because those models operate on
          whole datasets rather than stacked arrays. `hyp.Pipeline`
          additionally accepts manipulator and aligner names in a step
          list.) An unknown name raises a `ValueError` saying exactly
          this.
        - class: a scikit-learn style model class, instantiated with the
          spec's 'args'/'kwargs' (or defaults)
        - dict: the canonical {'model': <str, class, or instance>,
          'args': [...], 'kwargs': {...}} (both 'args' and 'kwargs'
          optional; 'args' are positional constructor arguments and
          require a name/class 'model'), or the legacy {'model': ...,
          'params': {...}} form (accepted for backward compatibility,
          emits a DeprecationWarning)
        - instance: any object exposing fit/transform/fit_transform/
          fit_predict/predict_proba (scikit-learn convention)
        - list: a pipeline; each element is applied in sequence, with each
          stage's output feeding the next stage's input

    mode : str
        How to apply the model: 'fit_transform', 'fit_predict',
        'predict_proba', or 'auto' (default), which prefers predict_proba
        (fitting first), then fit_transform/transform, then fit_predict.
        For a list (pipeline) spec, an explicit mode applies to the FINAL
        stage only (earlier stages run in 'auto' mode), so e.g.
        ['PCA', 'KMeans'] with mode='fit_predict' reduces, then returns
        cluster labels. An explicit mode the model cannot honor raises a
        ValueError naming the modes it does support. Note that 'auto' can
        return a different result KIND than hyp.Pipeline's fit_transform
        for the same model (e.g. GaussianMixture: membership probabilities
        here via predict_proba, hard labels from a raw Pipeline step).

    return_model : bool
        If True, also return the fitted model (default: False):

        - single spec + stack=True: the fitted model instance
        - single spec + stack=False: a LIST of fitted instances, one per
          dataset (even for a single input dataset)
        - list spec + stack=True: a fitted `hypertools.Pipeline` wrapping
          each stage's already-fitted model, in order
        - list spec + stack=False: a list of fitted `hypertools.Pipeline`
          objects, one per dataset

        Fitted models/pipelines are reusable on held-out data via
        `.transform()`/`pipeline.named_steps`/`pipeline.steps`. Caveat:
        models with no out-of-sample method (e.g. TSNE, MDS,
        SpectralEmbedding) cannot honor `.transform()` on new data --
        reusing such a pipeline raises a TypeError explaining this.

    format_data : bool
        Whether to run hypertools' format_data on the input (default: True).

    stack : bool
        If True (default), datasets are vertically stacked and the model is
        fit ONCE across all of them, then results are split back to match
        the input structure. NOTE: with stack=True, a passed-in model
        INSTANCE is fitted in place (the caller's object gains the fitted
        state). If False, a separate model is fit per dataset; each
        dataset gets its own clone and the caller's instance is left
        untouched.

    ndims : int or None
        Convenience: sets n_components on models that accept it. Must be a
        positive integer (or None); invalid values raise the same
        `ValueError` as `hyp.reduce`'s `ndims=` (release-1.0 audit,
        X2-error-quality-018: `apply_model(x, 'PCA', ndims=0)` used to
        silently return a width-0 array).

    Returns
    -------
    result (and fitted model(s) if return_model=True; see `return_model`
    for the exact shape). Lists in, lists out: a single input dataset
    returns a single result.
    """
    # ndims parity with hyp.reduce (release-1.0 audit, X2-error-quality-018:
    # reduce(ndims=0) raised while apply_model(..., ndims=0) silently
    # returned an (n, 0) array) -- same messages as reduce's validation.
    if ndims is not None:
        if isinstance(ndims, bool) or not isinstance(ndims, (int, np.integer)):
            raise ValueError(
                f"ndims must be a positive integer or None; got {ndims!r}")
        if ndims < 1:
            raise ValueError(f"ndims must be >= 1; got {ndims}")

    single_input = not isinstance(data, list)
    if format_data:
        # lazy: importing tools.format_data at module load time would create
        # a circular import (tools -> ..._shared.exceptions -> core.exceptions
        # -> core (package init) -> core.model -> tools), since core.model is
        # now exported from core/__init__.py.
        from ..tools.format_data import format_data as formatter
        data = formatter(data, ppca=True)
    elif single_input:
        data = [data]

    # pipeline: thread the data through each stage. `mode` describes the
    # requested OUTPUT, so it applies to the FINAL stage only; earlier
    # stages run in 'auto' mode (2026-07 audit F21-004: passing e.g.
    # mode='fit_predict' into every stage made reduce->cluster pipelines
    # crash on the first non-predicting stage).
    if isinstance(model, list):
        fitted = []
        last = len(model) - 1
        for i, stage in enumerate(model):
            data, stage_model = apply_model(
                data, stage, mode=(mode if i == last else 'auto'),
                return_model=True, format_data=False, stack=stack,
                ndims=ndims)
            if not isinstance(data, list):
                data = [data]
            fitted.append(stage_model)
        result = data if not single_input else data[0]
        if not return_model:
            return result
        # each `stage_model` is already a fitted scikit-learn-style
        # instance (not a class/dict/string spec), so Pipeline's step
        # resolution passes it through unchanged (never re-fit) --
        # constructing the Pipeline from already-fitted steps and marking
        # it fitted (skipping fit_transform, which would re-fit them) is
        # what lets `pipeline.transform(held_out)` reuse this exact fit
        from .pipeline import Pipeline
        if stack or not fitted:
            pipeline = Pipeline(fitted)
            pipeline._is_fitted = True
            return (result, pipeline)
        # stack=False: each stage fitted a SEPARATE model per dataset, so
        # hand back one fitted Pipeline per dataset (2026-07 audit F21-003:
        # this used to wrap the raw per-stage lists in a single Pipeline
        # whose .transform crashed with AttributeError)
        pipelines = []
        for j in range(len(fitted[0])):
            per_dataset = Pipeline([stage_models[j] for stage_models in fitted])
            per_dataset._is_fitted = True
            pipelines.append(per_dataset)
        return (result, pipelines)

    model_instance = _resolve_model(model, ndims)

    lengths = [np.asarray(d).shape[0] for d in data]
    arrays = [np.asarray(d, dtype=np.float64) for d in data]

    if stack:
        stacked = np.vstack(arrays)
        applied, fitted = _apply_single(model_instance, stacked, mode)
        pieces = np.vsplit(np.asarray(applied), np.cumsum(lengths)[:-1]) \
            if np.asarray(applied).ndim > 1 else \
            np.split(np.asarray(applied), np.cumsum(lengths)[:-1])
        result = [np.asarray(p) for p in pieces]
    else:
        result, fitted = [], []
        for arr in arrays:
            applied, fm = _apply_single(_clone(model_instance), arr, mode)
            result.append(np.asarray(applied))
            fitted.append(fm)

    if single_input:
        result = result[0]
    return (result, fitted) if return_model else result


def _resolve_model(model, ndims):
    """Turn a model specification into a ready-to-fit instance.

    Delegates dict/legacy-dict unpacking to `core.shared.unpack_model` (the
    single eval-free spec resolver every dispatcher shares) instead of
    re-implementing it here; only the string-registry lookup and the
    `ndims=` convenience are specific to `apply_model`.
    """
    from .shared import unpack_model
    resolved = unpack_model(model)

    args, params = [], {}
    if isinstance(resolved, dict):
        # {'model': ..., 'args': [...], 'kwargs': {...}} (the LEGACY
        # {'model', 'params'} form is normalized to this shape --  with a
        # DeprecationWarning -- by unpack_model itself)
        if 'model' not in resolved:
            raise ValueError(
                f"dict model specs require a 'model' key; got keys "
                f"{sorted(resolved)}. Use "
                "{'model': 'PCA', 'args': [...], 'kwargs': {...}} "
                "('args'/'kwargs' optional).")
        args = list(resolved.get('args', []))
        params = dict(resolved.get('kwargs', {}))
        resolved = resolved['model']

    if isinstance(resolved, str):
        registry = _build_registry()
        if resolved not in registry:
            raise ValueError(
                f'unknown model {resolved!r}; supported names: '
                f'{", ".join(supported_models())} (or pass a scikit-learn '
                f'style class or instance directly). apply_model covers the '
                f'REDUCE and CLUSTER model families only; manipulator, '
                f'aligner, forecaster, and imputer models have their own '
                f'dispatchers -- use hyp.manip, hyp.align, hyp.predict, or '
                f'hyp.impute for those (hyp.Pipeline also accepts '
                f"manipulator/aligner names such as 'ZScore' or "
                f"'HyperAlign' in a step list).")
        if resolved == 'UMAP':
            import warnings
            with warnings.catch_warnings():
                # umap-learn emits an ImportWarning about its OPTIONAL
                # tensorflow/ParametricUMAP extra at import time -- noise
                # for standard UMAP use (release-1.0 audit, X4-warnings-011;
                # same pattern as the gensim filter in tools/text2mat).
                warnings.filterwarnings(
                    'ignore', message='Tensorflow not installed')
                from umap import UMAP as model_cls
        else:
            model_cls = registry[resolved]
        resolved = model_cls

    if isinstance(resolved, type):
        # a registry name or a bare class: instantiate it with the spec's
        # 'args'/'kwargs' (2026-07 audit F21-001: 'args' used to be
        # silently dropped, so {'model': 'PCA', 'args': [3]} fit PCA with
        # its default components).
        # only inject n_components on models that actually accept it (QC 2026-07:
        # this used to force n_components onto e.g. KMeans -> "unexpected keyword
        # argument 'n_components'"; the instance path below already guards it).
        if (ndims is not None and not args and 'n_components' not in params
                and 'n_components' in inspect.signature(resolved).parameters):
            params['n_components'] = ndims
        return resolved(*args, **params)

    # instance: duck-type on the sklearn convention
    if not any(hasattr(resolved, m) for m in
               ('fit_transform', 'fit_predict', 'transform', 'predict')):
        raise ValueError(
            'model instances must follow the scikit-learn convention '
            '(fit/transform/fit_transform/fit_predict/predict_proba)')
    if args:
        raise ValueError(
            f"'args' ({args!r}) cannot be applied to an already-constructed "
            f"{type(resolved).__name__} instance: positional constructor "
            "arguments require a class or registry-name spec (e.g. "
            "{'model': 'PCA', 'args': [3]}); for an instance, move the "
            "values into 'kwargs' or construct the instance with them "
            "directly.")
    if params:
        resolved.set_params(**params)
    if ndims is not None and hasattr(resolved, 'n_components'):
        resolved.set_params(n_components=ndims)
    return resolved


def _supported_modes(model):
    """The explicit `mode=` values this model can honor."""
    modes = []
    if hasattr(model, 'fit_transform') or (hasattr(model, 'fit')
                                           and hasattr(model, 'transform')):
        modes.append('fit_transform')
    if hasattr(model, 'fit_predict') or (hasattr(model, 'fit')
                                         and hasattr(model, 'predict')):
        modes.append('fit_predict')
    if hasattr(model, 'predict_proba'):
        modes.append('predict_proba')
    return modes


def _apply_single(model, stacked, mode):
    """Fit and apply one model to one (stacked) array.

    An explicit `mode=` the model cannot honor raises a clear ValueError
    naming the model, the mode, and the modes it does support (2026-07
    audit F21-004: mismatches used to surface as raw AttributeErrors like
    "'PCA' object has no attribute 'predict_proba'").
    """
    if mode == 'auto':
        if hasattr(model, 'predict_proba'):
            mode = 'predict_proba'
        elif hasattr(model, 'fit_transform') or hasattr(model, 'transform'):
            mode = 'fit_transform'
        else:
            mode = 'fit_predict'
    elif mode in ('fit_transform', 'fit_predict', 'predict_proba'):
        supported = _supported_modes(model)
        if mode not in supported:
            raise ValueError(
                f"{type(model).__name__} does not support mode={mode!r} "
                f"(it has no matching method); this model supports: "
                f"{', '.join(supported) or 'none of the explicit modes'}. "
                f"Use one of those, or mode='auto'.")

    if mode == 'fit_transform':
        if hasattr(model, 'fit_transform'):
            return model.fit_transform(stacked), model
        model.fit(stacked)
        return model.transform(stacked), model
    if mode == 'predict_proba':
        model.fit(stacked)
        return model.predict_proba(stacked), model
    if mode == 'fit_predict':
        if hasattr(model, 'fit_predict'):
            return model.fit_predict(stacked), model
        model.fit(stacked)
        return model.predict(stacked), model
    raise ValueError(f'unknown mode {mode!r}; use fit_transform, '
                     f'fit_predict, predict_proba, or auto')


def _clone(model):
    try:
        from sklearn.base import clone
        return clone(model)
    except Exception:
        return model
