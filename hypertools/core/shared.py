"""Shared core helpers: a forgiving dict and an eval-free model-spec resolver.

`unpack_model` is the eval-free replacement for the fork's string→eval model
lookup: names are matched against an explicit whitelist of classes, objects are
checked against a parent class, and dict specs have their inner model unpacked
recursively. Anything unmatched is returned unchanged for the registry to
resolve later.
"""
import warnings


def is_reused_pipeline(spec, stage_kwargs, spec_label):
    """Whether `spec` is a whole already-fitted `hypertools.Pipeline` handed
    back as a dispatcher's model spec (e.g. the model returned by an earlier
    cross-module `return_model=True` call, such as `hyp.cluster(..., reduce=,
    manip=, return_model=True)`).

    When it is, the caller should REUSE it via `spec.transform(data)` rather
    than trying to wrap it in a Reducer/Clusterer/Aligner (which used to crash
    with e.g. ``AttributeError: 'Pipeline' object has no attribute 'labels_'``,
    QC 2026-07) -- the fitted Pipeline already encodes its own manip/normalize/
    reduce/align/cluster stages. Any of those stages re-specified alongside it
    would double-apply, so they are reported as redundant and ignored.

    Parameters
    ----------
    spec : object
        The primary model spec passed to the dispatcher (`reduce=`/`cluster=`/
        `model=`).
    stage_kwargs : dict
        `{name: value}` for the OTHER cross-module stage kwargs on that
        dispatcher (used only to warn about redundant ones).
    spec_label : str
        The dispatcher's spec-kwarg name (`'reduce'`, `'cluster'`, `'model'`),
        for the warning message.

    Returns
    -------
    bool
        True if `spec` is a fitted `Pipeline` to reuse via `.transform`.
    """
    from .pipeline import Pipeline
    if isinstance(spec, Pipeline) and spec.is_fitted:
        redundant = sorted(name for name, value in stage_kwargs.items()
                           if value is not None)
        if redundant:
            warnings.warn(
                f"{spec_label}= is an already-fitted Pipeline that encodes its "
                f"own stages; ignoring redundant {', '.join(redundant)}= (the "
                "fitted Pipeline is reused as-is via .transform).",
                stacklevel=3)
        return True
    return False


class RobustDict(dict):
    """dict whose missing keys return a default value instead of raising."""

    def __init__(self, *args, **kwargs):
        self.default_value = kwargs.pop("__default_value__", None)
        super().__init__(*args, **kwargs)

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            return self.default_value

    def __missing__(self, key):
        return self.default_value


def unpack_model(m, valid=None, parent_class=None):
    """Resolve a model specification without eval.

    Parameters
    ----------
    m : str, class, instance, dict, or list of these
        - str: matched against ``valid``'s class names (eval-free lookup)
        - class or instance: matched against ``parent_class``; any other
          (non-dict, non-list, non-string) object is treated as an
          already-constructed/already-fitted model and passed through
          unchanged, so a fitted instance handed back in (e.g. from an
          earlier ``return_model=True`` call) is never mistaken for a spec
        - dict ``{'model': ..., 'args': [...], 'kwargs': {...}}``: the
          canonical dict form; ``'model'`` is unpacked recursively
        - dict ``{'model': ..., 'params': {...}}``: the LEGACY dict form
          (dev-1.0/fork). Accepted for backward compatibility, but emits a
          ``DeprecationWarning`` and is translated to the canonical
          ``{'model': ..., 'args': [], 'kwargs': params}`` shape before
          unpacking, so every caller of ``unpack_model`` inherits the
          warning for free
        - list: mapped element-wise (a pipeline of specs)

    valid : list of classes whose ``__name__`` a string may match
    parent_class : if given, an ``m`` that is/instantiates a subclass passes through

    Returns
    -------
    The matched class, the object itself, a dict with its inner ``'model'``
    unpacked, or (for an unmatched string) the string unchanged.
    """
    if isinstance(m, list):
        return [unpack_model(x, valid=valid, parent_class=parent_class) for x in m]

    if valid is None:
        valid = []

    if isinstance(m, str) and m in [v.__name__ for v in valid]:
        return next(v for v in valid if v.__name__ == m)

    if parent_class is not None:
        try:
            if issubclass(m, parent_class):
                return m
        except TypeError:
            if isinstance(m, parent_class):
                return m

    if isinstance(m, dict):
        if "model" in m and "params" in m and "args" not in m and "kwargs" not in m:
            warnings.warn(
                "{'model': ..., 'params': {...}} is deprecated; use "
                "{'model': ..., 'args': [...], 'kwargs': {...}} instead",
                DeprecationWarning, stacklevel=2)
            m = {"model": m["model"], "args": [], "kwargs": dict(m["params"])}

        # canonical dict spec: a 'model' key with OPTIONAL 'args'/'kwargs'
        # (either may be omitted -- e.g. {'model': 'Smooth', 'kwargs': {...}}
        # or a bare {'model': 'PCA'} -- matching reduce/cluster, which accept
        # `'args' in x or 'kwargs' in x`). Missing pieces default to []/{}.
        if "model" in m:
            resolved = dict(m)
            resolved["model"] = unpack_model(m["model"], valid=valid, parent_class=parent_class)
            resolved.setdefault("args", [])
            resolved.setdefault("kwargs", {})
            return resolved

    if isinstance(m, str):
        return m

    if isinstance(m, type):
        raise ValueError(f"unknown model: {m!r}")

    # anything else (an already-constructed or already-fitted instance) is
    # passed through unchanged for the caller to duck-type/apply directly --
    # but only when no parent_class was given to validate against (the case
    # Pipeline._resolve_step needs, e.g. a fitted sklearn model handed back
    # in from an earlier return_model=True call). When a parent_class WAS
    # given and this object didn't match it above, it is not a valid spec.
    if parent_class is None:
        return m

    raise ValueError(f"unknown model: {m!r}")


def get(value, i):
    """Return value[i] for a list/tuple (if in range), else value itself.

    Lets a manipulator accept either one shared parameter or a per-dataset list.
    """
    if isinstance(value, (list, tuple)):
        if 0 <= i < len(value):
            return value[i]
        return value
    return value
