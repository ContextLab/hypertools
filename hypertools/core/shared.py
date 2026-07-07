"""Shared core helpers: a forgiving dict and an eval-free model-spec resolver.

`unpack_model` is the eval-free replacement for the fork's string→eval model
lookup: names are matched against an explicit whitelist of classes, objects are
checked against a parent class, and dict specs have their inner model unpacked
recursively. Anything unmatched is returned unchanged for the registry to
resolve later.
"""
import warnings


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

        if all(k in m for k in ("model", "args", "kwargs")):
            resolved = dict(m)
            resolved["model"] = unpack_model(m["model"], valid=valid, parent_class=parent_class)
            return resolved

    if isinstance(m, str):
        return m

    if isinstance(m, type):
        raise ValueError(f"unknown model: {m!r}")

    # anything else (an already-constructed or already-fitted instance) is
    # passed through unchanged for the caller to duck-type/apply directly
    return m


def get(value, i):
    """Return value[i] for a list/tuple (if in range), else value itself.

    Lets a manipulator accept either one shared parameter or a per-dataset list.
    """
    if isinstance(value, (list, tuple)):
        if 0 <= i < len(value):
            return value[i]
        return value
    return value
