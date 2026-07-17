"""Shared core helpers: a forgiving dict and an eval-free model-spec resolver.

`unpack_model` is the eval-free replacement for the fork's string→eval model
lookup: names are matched against an explicit whitelist of classes, objects are
checked against a parent class, and dict specs have their inner model unpacked
recursively. Anything unmatched is returned unchanged for the registry to
resolve later.
"""
import copy
import warnings

#: sentinel distinguishing "no explicit default passed" in RobustDict.get
_MISSING = object()


def require_data(data, caller):
    """Reject ``data=None`` with the ONE unified error every dispatcher uses.

    2026-07 release audit (final wave, item 9): the dispatchers used to
    disagree about `None` input -- `manip`/`impute`/`predict` raised
    slightly different `TypeError`\\ s, `align` raised a misleading
    ``ValueError: input has no observations (0 rows)`` from deep inside
    `format_data`, and `analyze(None)` silently returned `None`. Every
    dispatcher now calls this first, so `None` always raises the same
    `TypeError` naming the entry point.

    Parameters
    ----------
    data : object
        The dispatcher's raw ``data`` argument.
    caller : str
        The public entry-point name (e.g. ``'align'``), for the message.

    Raises
    ------
    TypeError
        If `data` is `None`.
    """
    if data is None:
        raise TypeError(
            f'Unsupported data type passed to {caller}: None. Supported '
            'types: Numpy Array, Pandas DataFrame/Series, String, List of '
            'strings, List of numbers, or a list of datasets.')


def no_observations_message(action, detail='0 rows'):
    """The ONE unified empty-input message every dispatcher raises.

    2026-07 release audit (final wave, item 10): empty inputs used to get
    five different phrasings ('input has no observations (0 rows)...',
    'cannot align an empty list...', 'cannot forecast an empty dataset...',
    ...). Everything now uses this template -- the same shape as
    `format_data`'s message, which several modules already matched.

    Parameters
    ----------
    action : str
        The verb for "there is nothing to <action>." (e.g. ``'align'``).
    detail : str
        What was actually received (e.g. ``'got an empty list'``,
        ``'dataset 1 has shape (0, 2)'``; default: ``'0 rows'``).

    Returns
    -------
    str
        ``'input has no observations (<detail>); there is nothing to
        <action>.'``
    """
    return (f'input has no observations ({detail}); there is nothing to '
            f'{action}.')


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
    """dict whose missing keys return a default value instead of raising.

    Consistency guarantees (2026-07 audit, F23-core-config-exceptions-004):

    - ``rd[key]``, ``rd.get(key)``, and ``rd.get(key, explicit)`` agree:
      the first two return the configured default for a missing key, the
      third returns the explicitly-passed one
    - every missing-key lookup returns a FRESH (deep) copy of the default,
      so mutating one result never pollutes later lookups
    - ``rd.copy()`` returns another ``RobustDict`` with the same default
      (plain ``dict(rd)`` still strips the default, as for any dict
      subclass)
    """

    def __init__(self, *args, **kwargs):
        self.default_value = kwargs.pop("__default_value__", None)
        super().__init__(*args, **kwargs)

    def _fresh_default(self):
        return copy.deepcopy(self.default_value)

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            return self._fresh_default()

    def __missing__(self, key):
        return self._fresh_default()

    def get(self, key, default=_MISSING):
        """Like dict.get, but a missing key falls back to this dict's
        configured ``__default_value__`` when no explicit ``default`` is
        given (so ``rd.get(k)`` and ``rd[k]`` agree)."""
        if key in self:
            return super().__getitem__(key)
        if default is not _MISSING:
            return default
        return self._fresh_default()

    def copy(self):
        """Return a shallow copy that is still a RobustDict (dict.copy
        used to silently degrade to a plain dict, dropping the default)."""
        return RobustDict(self, __default_value__=self.default_value)


def unpack_model(m, valid=None, parent_class=None):
    """Resolve a model specification without eval.

    Parameters
    ----------
    m : str, class, instance, dict, or list of these
        - str: matched against ``valid``'s class names (eval-free lookup)
        - class or instance: matched against ``parent_class``; when no
          ``parent_class`` is given, any other (non-dict, non-list,
          non-string) object -- a bare class for the caller to
          instantiate, or an already-constructed/already-fitted model --
          is passed through unchanged, so a fitted instance handed back in
          (e.g. from an earlier ``return_model=True`` call) is never
          mistaken for a spec
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
            if "params" in resolved:
                # legacy 'params' alongside canonical 'args'/'kwargs' (2026-07
                # audit F23-007): it used to be retained inertly and silently
                # ignored downstream -- warn and drop it so the mistake is
                # visible (the canonical keys win)
                dropped = resolved.pop("params")
                warnings.warn(
                    f"ignoring the legacy 'params' key ({dropped!r}) because "
                    "'args'/'kwargs' are also present in the model spec; "
                    "merge those values into 'kwargs' instead",
                    DeprecationWarning, stacklevel=2)
            resolved["model"] = unpack_model(m["model"], valid=valid, parent_class=parent_class)
            resolved.setdefault("args", [])
            resolved.setdefault("kwargs", {})
            return resolved

    if isinstance(m, str):
        return m

    if isinstance(m, type):
        # mirror the instance behavior below: with no parent_class to
        # validate against, a bare class is a legitimate spec for the
        # caller to instantiate (2026-07 audit F23-008/F21-007: classes
        # used to raise here while instances passed through)
        if parent_class is None:
            return m
        raise ValueError(
            f"unknown model spec: class {m.__name__} is not a subclass of "
            f"{parent_class.__name__}. Pass "
            + (f"one of {', '.join(sorted(v.__name__ for v in valid))}; "
               if valid else "")
            + f"a {parent_class.__name__} subclass or instance; or a dict "
            "spec {'model': ..., 'args': [...], 'kwargs': {...}}.")

    # anything else (an already-constructed or already-fitted instance) is
    # passed through unchanged for the caller to duck-type/apply directly --
    # but only when no parent_class was given to validate against (the case
    # Pipeline._resolve_step needs, e.g. a fitted sklearn model handed back
    # in from an earlier return_model=True call). When a parent_class WAS
    # given and this object didn't match it above, it is not a valid spec.
    if parent_class is None:
        return m

    raise ValueError(
        f"unknown model spec: got a {type(m).__name__} instance, which is "
        f"not a {parent_class.__name__} (or subclass) instance. Pass "
        + (f"one of {', '.join(sorted(v.__name__ for v in valid))}; "
           if valid else "")
        + f"a {parent_class.__name__} subclass or instance; or a dict spec "
        "{'model': ..., 'args': [...], 'kwargs': {...}}.")


def get(value, i):
    """Return value[i] for a list/tuple (if in range), else value itself.

    Lets a manipulator accept either one shared parameter or a per-dataset
    list. Negative indices follow the usual Python convention
    (``get(v, -1)`` is ``v[-1]``). An out-of-range index means a
    per-dataset list was shorter than the number of datasets: the whole
    list is still returned (the historical broadcast behavior), but a
    ``UserWarning`` now flags the length mismatch (2026-07 audit,
    F23-core-config-exceptions-012).
    """
    if isinstance(value, (list, tuple)):
        n = len(value)
        if -n <= i < n:
            return value[i]
        warnings.warn(
            f"parameter list of length {n} has no entry for dataset index "
            f"{i}; using the whole list as this dataset's value. Pass a "
            "scalar to share one value across all datasets, or a list with "
            "one entry per dataset.", stacklevel=2)
        return value
    return value
