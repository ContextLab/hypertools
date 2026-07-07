"""Unified spec resolution + the `hyp.Pipeline` class.

`Pipeline` chains hypertools model specs (strings, classes, instances, dict
specs, or nested pipelines -- the grammar in `unpack_model`) the same way
scikit-learn's `Pipeline` chains estimators: `fit`/`fit_transform` re-fits
every step from scratch; `transform` re-applies steps that were already
fit(-transformed), never refitting them.

`build_pipeline` is the helper every dispatcher (manip/normalize/reduce/
align/cluster) uses to assemble the cross-module stage kwargs (`manip=`,
`normalize=`, `reduce=`, `align=`, `cluster=`) into a `Pipeline` in the
canonical order (#153). Each stage is wrapped as a `_DispatchStep`: it
fits by calling the stage's own dispatcher (`hyp.manip`/`hyp.normalize`/
`hyp.reduce`/`hyp.align`/`hyp.cluster`) with `return_model=True` and the
ORIGINAL spec, then reuses the fitted model on `transform` by calling the
SAME dispatcher again with the FITTED model handed back in as the spec --
every dispatcher already documents (as the payoff of its own
`return_model=True`) that a previously-fitted wrapper passed back in as
the spec kwarg is applied via `.transform`, never refit. This keeps each
dispatcher's own list<->stacked-array shape handling (`reduce`/`cluster`
stack every dataset into one array internally; `manip`/`normalize`/`align`
operate on lists directly) rather than requiring `Pipeline.transform` to
know about it, while still genuinely avoiding a refit on `transform`
(round17 Task 6 fix: the previous `_CallableStep` re-ran each dispatcher
with the ORIGINAL, unfitted spec on every call, silently refitting on
every `.transform()` -- e.g. a `reduce='PCA'` stage would fit a brand
NEW PCA basis on whatever data `.transform` was given, rather than
reusing the basis fit the first time).
"""
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

    # `Aligner.fit`/`fit_transform`/`transform` (align/common.py) are
    # already genuinely fit/transform-capable -- `transform(new_data)`
    # validates `new_data`'s shape against the fit-time shape (#227) and
    # applies the fitted alignment to it -- so no wrapper is needed here
    # (unlike pre-#227, when `transform` ignored its argument entirely).
    return model


def _base_name(model):
    """Auto-name a step the way scikit-learn does: the lowercased class
    name of the underlying model."""
    if isinstance(model, Pipeline):
        return 'pipeline'
    return type(model).__name__.lower()


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


class _DispatchStep:
    """Wrap a stage dispatcher (`hyp.manip`/`hyp.normalize`/`hyp.reduce`/
    `hyp.align`/`hyp.cluster`) as a fit/transform step that genuinely
    reuses its fitted model on `transform`, rather than re-running the
    dispatcher with the ORIGINAL (possibly unfitted) spec every time.

    `fit_transform` calls the dispatcher with `return_model=True` and the
    spec `build_pipeline` was given; `transform` calls the SAME dispatcher
    again, but with the FITTED model (returned by the previous call)
    substituted in as the spec -- reusing each dispatcher's own
    already-fitted-instance branch (documented as the payoff of
    `return_model=True` on every one of them) instead of refitting.
    """
    def __init__(self, name, spec, call):
        self._name = name
        self._spec = spec
        self._call = call  # (data, spec_or_fitted_model) -> (result, fitted_model)
        self._fitted = None

    @property
    def is_fitted(self):
        """Whether `fit`/`fit_transform` has been called on this step yet."""
        return self._fitted is not None

    def fit(self, data):
        """Fit this step's dispatcher on `data`, discarding the transformed result.

        Parameters
        ----------
        data : the data to fit on.

        Returns
        -------
        _DispatchStep
            `self`, for chaining (mirrors scikit-learn's `fit` convention).
        """
        self.fit_transform(data)
        return self

    def fit_transform(self, data):
        """Fit this step's dispatcher on `data` and return the transformed result.

        Calls the wrapped dispatcher with `return_model=True` and the
        original spec, storing the returned fitted model for later
        `transform` calls.

        Parameters
        ----------
        data : the data to fit and transform.

        Returns
        -------
        The dispatcher's transformed output for `data`.
        """
        result, fitted = self._call(data, self._spec)
        self._fitted = fitted
        return result

    def transform(self, data):
        """Apply the already-fitted dispatcher model to new `data`.

        Parameters
        ----------
        data : the data to transform, using the model fitted by a prior
            `fit`/`fit_transform` call.

        Returns
        -------
        The dispatcher's transformed output for `data`.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If `fit`/`fit_transform` has not been called yet.
        """
        if self._fitted is None:
            raise NotFittedError(f'{self._name} stage must be fit before transform')
        result, fitted = self._call(data, self._fitted)
        self._fitted = fitted
        return result

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

    Notes
    -----
    `steps` stores already-*resolved* `(name, instance)` tuples (specs are
    resolved to instances in `__init__`), not the raw constructor input. This
    deviates from scikit-learn's clone contract, which expects `get_params`/
    `set_params` to round-trip the exact constructor arguments -- so
    `sklearn.base.clone(pipe)` compatibility is not guaranteed.

    Examples
    --------
    >>> from hypertools import Pipeline
    >>> pipe = Pipeline(['ZScore', 'PCA'])  # doctest: +SKIP
    >>> out = pipe.fit_transform(x)  # doctest: +SKIP
    >>> out2 = pipe.transform(other_x)  # doctest: +SKIP
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
        return _DispatchStep(
            'manip', spec,
            lambda data, m: _manip(data, model=m, return_model=True))
    if stage == 'normalize':
        from ..tools.normalize import normalize as _normalize
        return _DispatchStep(
            'normalize', spec,
            lambda data, m: _normalize(data, normalize=m, return_model=True))
    if stage == 'reduce':
        from ..reduce.reduce import reduce as _reduce
        return _DispatchStep(
            'reduce', spec,
            lambda data, m: _reduce(data, reduce=m, ndims=ndims, return_model=True))
    if stage == 'align':
        from ..align.align import align as _align
        return _DispatchStep(
            'align', spec,
            lambda data, m: _align(data, model=m, return_model=True))
    if stage == 'cluster':
        from ..cluster.cluster import cluster as _cluster
        return _DispatchStep(
            'cluster', spec,
            lambda data, m: _cluster(data, cluster=m, return_model=True))
    raise ValueError(f"unknown pipeline stage {stage!r}; expected one of {CANONICAL_ORDER}")
