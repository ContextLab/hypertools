"""Unified spec resolution + the `hyp.Pipeline` class.

`Pipeline` chains hypertools model specs (strings, classes, instances, dict
specs, or nested pipelines -- the grammar in `unpack_model`) the same way
scikit-learn's `Pipeline` chains estimators: `fit`/`fit_transform` re-fits
every step from scratch; `transform` re-applies steps that were already
fit(-transformed), never refitting them.

`build_pipeline` is the helper every dispatcher (manip/normalize/reduce/
align/cluster) will use (Tasks 2-6) to assemble the cross-module stage
kwargs (`manip=`, `normalize=`, `reduce=`, `align=`, `cluster=`) into a
`Pipeline` in the canonical order (#153). reduce/cluster/normalize are still
plain (pre-1.0) functions rather than fit/transform-capable classes, so
their steps are wrapped as thin re-run-on-call adapters here; manip/align
already have class-based dispatchers and get the same functional wrapping
for interface consistency -- Tasks 2/3 swap these wrappers for genuine
fitted-model reuse once `return_model` lands on every dispatcher, without
changing `build_pipeline`'s signature or call sites.
"""
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError

from .shared import unpack_model


#: the single documented pipeline order (#153): impute happens during
#: load/format, upstream of everything here.
CANONICAL_ORDER = ('manip', 'normalize', 'reduce', 'align', 'cluster')


def _resolve_step_class(name):
    """Look up a bare string spec across every stage's registry."""
    if name == 'UMAP':
        from umap import UMAP
        return UMAP

    from ..manip.manip import MANIPULATORS
    from ..align.align import ALIGNERS
    from ..reduce.reduce import models as reduce_models
    from ..cluster.cluster import models as cluster_models, mixture_models as cluster_mixture_models

    registry = {}
    registry.update(reduce_models)
    registry.update(cluster_models)
    registry.update(cluster_mixture_models)
    for cls in list(MANIPULATORS) + list(ALIGNERS):
        registry[cls.__name__] = cls

    if name not in registry:
        raise ValueError(
            f"unknown pipeline step {name!r}; supported names: "
            f"{', '.join(sorted(registry) + ['UMAP'])}")
    return registry[name]


def _resolve_ref(ref):
    """A string becomes a class (via the combined registry); anything else
    (a class or an instance) passes through unchanged."""
    if isinstance(ref, str):
        return _resolve_step_class(ref)
    return ref


def _resolve_step(spec):
    """Turn a bare model spec (str/class/instance/dict) into a fit/transform
    -capable object. Fitted instances, callables, and nested Pipelines pass
    through unchanged."""
    if isinstance(spec, Pipeline):
        return spec

    if isinstance(spec, dict):
        resolved = unpack_model(spec, valid=[], parent_class=None)
        if isinstance(resolved, dict):
            inner = _resolve_ref(resolved['model'])
            args = resolved.get('args', [])
            kwargs = resolved.get('kwargs', {})
        else:
            inner = _resolve_ref(resolved)
            args, kwargs = [], {}
        if isinstance(inner, type):
            model = inner(*args, **kwargs)
        else:
            if kwargs and hasattr(inner, 'set_params'):
                inner.set_params(**kwargs)
            model = inner
    else:
        ref = _resolve_ref(spec)
        model = ref() if isinstance(ref, type) else ref

    from ..align.common import Aligner
    if isinstance(model, Aligner):
        return _AlignedStep(model)
    return model


def _base_name(model):
    """Auto-name a step the way scikit-learn does: the lowercased class
    name of the (unwrapped) underlying model."""
    if isinstance(model, Pipeline):
        return 'pipeline'
    target = getattr(model, '_aligner', model)
    return type(target).__name__.lower()


def _name_steps(entries):
    """entries: list of (explicit_name_or_None, model). Fills in auto names
    for entries without one, resolving collisions with a numeric suffix
    (e.g. two unnamed HyperAlign steps become 'hyperalign', 'hyperalign-1')."""
    used = {name for name, _ in entries if name is not None}
    counts = {}
    named = []
    for name, model in entries:
        if name is None:
            base = _base_name(model)
            n = counts.get(base, 0)
            candidate = base
            while candidate in used:
                n += 1
                candidate = f"{base}-{n}"
            counts[base] = n
            name = candidate
        used.add(name)
        named.append((name, model))
    return named


class _AlignedStep:
    """Wrap an Aligner so a Pipeline can validate/apply it to NEW data.

    `Aligner.transform` (align/common.py) ignores the argument passed to it
    and always replays the data it was fit on -- there is no notion of
    "apply this fitted alignment to a different dataset". This wrapper
    records the fit-time dataset count and per-dataset column counts and,
    on `transform`, validates new data against them (raising `ValueError`
    naming the fit-time shape -- #227) before temporarily pointing the
    aligner at the new data to compute a genuine projection of it.
    """
    def __init__(self, aligner):
        self._aligner = aligner
        self._n_datasets = None
        self._n_columns = None

    @staticmethod
    def _shape_of(data):
        items = data if isinstance(data, list) else [data]
        return len(items), [np.asarray(d).shape[1] for d in items]

    def fit(self, data):
        self._aligner.fit(data)
        self._n_datasets, self._n_columns = self._shape_of(data)
        return self

    def fit_transform(self, data):
        out = self._aligner.fit_transform(data)
        self._n_datasets, self._n_columns = self._shape_of(data)
        return out

    def transform(self, data):
        if self._n_datasets is None:
            raise NotFittedError('must fit align step before transforming data')
        n_datasets, n_columns = self._shape_of(data)
        if n_datasets != self._n_datasets or n_columns != self._n_columns:
            raise ValueError(
                f"align step was fit on {self._n_datasets} dataset(s) with "
                f"{self._n_columns} column(s) each; got {n_datasets} "
                f"dataset(s) with {n_columns} column(s) (fit-time shape: "
                f"{self._n_datasets} datasets x {self._n_columns} columns)")
        original_data = self._aligner.data
        self._aligner.data = data
        try:
            return self._aligner.transform()
        finally:
            self._aligner.data = original_data

    def __repr__(self):
        return f"AlignStep({self._aligner!r})"


class _CallableStep:
    """Wrap a plain `data -> result` callable (a pre-1.0 dispatcher function
    bound to one spec) as a fit/transform step. `reduce`/`cluster`/
    `normalize` (and, until Tasks 2/3 give them `return_model`, `manip`/
    `align` too) have no persisted fitted state to reuse, so `transform`
    re-runs the same call rather than genuinely reusing a fit."""
    def __init__(self, name, call):
        self._name = name
        self._call = call

    def fit(self, data):
        self._call(data)
        return self

    def fit_transform(self, data):
        return self._call(data)

    def transform(self, data):
        return self._call(data)

    def __repr__(self):
        return f"<{self._name} stage>"


class Pipeline(BaseEstimator):
    """Chain hypertools model specs into one fit/transform-able object.

    Mirrors scikit-learn's `Pipeline`: `fit`/`fit_transform` fit every step
    from scratch (in order); `transform` re-applies the already-fitted
    steps to new data without refitting them.

    Parameters
    ----------
    steps : list
        Each element is either a `(name, spec)` tuple or a bare spec. A
        spec is anything `unpack_model` accepts: a registry name (string),
        a class, an already-constructed (or already-fitted) instance, a
        dict spec (`{'model': ..., 'args': [...], 'kwargs': {...}}` or the
        legacy `{'model': ..., 'params': {...}}`), or a nested `Pipeline`.
        Bare specs are auto-named after their resolved class (lowercased),
        with a numeric suffix on collision (`'hyperalign'`, then
        `'hyperalign-1'`).

    Attributes
    ----------
    steps : list of (str, object)
        The named, resolved (but not-yet-necessarily-fitted) steps, in
        order.

    Examples
    --------
    >>> from hypertools import Pipeline
    >>> pipe = Pipeline(['ZScore', 'PCA'])
    >>> out = pipe.fit_transform(x)  # doctest: +SKIP
    >>> out2 = pipe.transform(other_x)  # reuses the fitted PCA/ZScore
    """

    def __init__(self, steps):
        entries = []
        for step in steps:
            if isinstance(step, tuple) and len(step) == 2 and isinstance(step[0], str):
                name, spec = step
            else:
                name, spec = None, step
            entries.append((name, _resolve_step(spec)))
        self.steps = _name_steps(entries)
        self._is_fitted = False

    @property
    def named_steps(self):
        """dict view of `self.steps`, keyed by step name."""
        return dict(self.steps)

    @property
    def is_fitted(self):
        """True once `fit`/`fit_transform` has been called at least once."""
        return self._is_fitted

    def fit(self, data):
        """Fit every step in order (see `fit_transform`); returns `self`."""
        self.fit_transform(data)
        return self

    def fit_transform(self, data):
        """Fit and apply every step in order, feeding each step's output to
        the next. Refits every step, even if some were already fitted."""
        out = data
        for _, model in self.steps:
            out = model.fit_transform(out)
        self._is_fitted = True
        return out

    def transform(self, data):
        """Apply the already-fitted steps (in order) to `data` without
        refitting them."""
        if not self._is_fitted:
            raise NotFittedError('Pipeline must be fit before transform')
        out = data
        for _, model in self.steps:
            out = model.transform(out)
        return out

    def inverse_transform(self, data):
        """Best-effort reverse pass through steps that implement
        `inverse_transform`, most-recent-step-first.

        Raises
        ------
        NotImplementedError
            Naming the first (in reverse order) step that has no
            `inverse_transform`.
        """
        out = data
        for name, model in reversed(self.steps):
            if not hasattr(model, 'inverse_transform'):
                raise NotImplementedError(
                    f"cannot inverse_transform through step {name!r} "
                    f"({type(model).__name__} has no inverse_transform)")
            out = model.inverse_transform(out)
        return out

    def __repr__(self):
        inner = ', '.join(f"{name}={model!r}" for name, model in self.steps)
        return f"Pipeline([{inner}])"


def build_pipeline(manip=None, normalize=None, reduce=None, ndims=None,
                    align=None, cluster=None, order=CANONICAL_ORDER):
    """Assemble a `Pipeline` from the cross-module stage kwargs (#138), in
    canonical order (#153).

    Each dispatcher in Tasks 2-6 will call this to build the `Pipeline` it
    hands back when `return_model=True` and more than one stage ran. reduce/
    cluster/normalize are called via their current (pre-1.0) public
    functions -- lazily imported here to avoid import cycles -- so their
    steps only support fit_transform-style reuse until those dispatchers
    gain a real `return_model`; `build_pipeline`'s own signature will not
    change when that happens.

    Parameters
    ----------
    manip, normalize, reduce, align, cluster : model spec or None
        A spec for the given stage (anything `unpack_model` accepts), or
        `None` to omit that stage entirely.

    ndims : int or None
        Passed through to the `reduce` stage as `ndims=`.

    order : tuple of str
        Stage names in the order they should run (default: `CANONICAL_ORDER`
        = `('manip', 'normalize', 'reduce', 'align', 'cluster')`).

    Returns
    -------
    Pipeline
        An (unfitted) `Pipeline` with one step per non-`None` stage kwarg,
        in `order`.
    """
    specs = {'manip': manip, 'normalize': normalize, 'reduce': reduce,
             'align': align, 'cluster': cluster}

    steps = []
    for stage in order:
        spec = specs.get(stage)
        if spec is None:
            continue
        steps.append((stage, _make_stage_step(stage, spec, ndims)))
    return Pipeline(steps)


def _make_stage_step(stage, spec, ndims):
    if stage == 'manip':
        from ..manip.manip import manip as _manip
        return _CallableStep('manip', lambda data: _manip(data, model=spec))
    if stage == 'normalize':
        from ..tools.normalize import normalize as _normalize
        return _CallableStep('normalize', lambda data: _normalize(data, normalize=spec))
    if stage == 'reduce':
        from ..reduce.reduce import reduce as _reduce
        return _CallableStep('reduce', lambda data: _reduce(data, reduce=spec, ndims=ndims))
    if stage == 'align':
        from ..align.align import align as _align
        return _CallableStep('align', lambda data: _align(data, model=spec))
    if stage == 'cluster':
        from ..cluster.cluster import cluster as _cluster
        return _CallableStep('cluster', lambda data: _cluster(data, cluster=spec))
    raise ValueError(f"unknown pipeline stage {stage!r}; expected one of {CANONICAL_ORDER}")
