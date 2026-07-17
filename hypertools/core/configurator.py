"""Published default options for hypertools, parsed from core/config.ini.

The dispatcher functions (``hyp.plot``/``reduce``/``align``/``cluster``/...)
define their defaults in their own signatures; ``core/config.ini`` MIRRORS
those defaults so they can be inspected programmatically through
``get_default_options()``/``apply_defaults()`` (and the agreement between the
two is enforced by tests). Editing config.ini does NOT change runtime
behavior -- it is a published, queryable record of the defaults, not the
mechanism that sets them.

Values are parsed with ``ast.literal_eval`` where possible, so numbers,
booleans, and ``None`` come back as real Python values (``3``, ``False``,
``None``) rather than raw INI strings; anything that is not a Python literal
(e.g. ``IncrementalPCA``) stays a string. Lookups for unconfigured sections
yield ``{}`` instead of raising (RobustDict).

datawrangler's defaults are merged in underneath hypertools' own, so both
libraries can be queried through one call.
"""
import ast
import os

# datawrangler 0.5.0 evaluated ``os.getenv('HOME')`` at import time to build
# its data directory; ``HOME`` is unset on Windows (which uses ``USERPROFILE``),
# so ``os.path.join(None, ...)`` raised ``TypeError`` and importing dw -- and
# therefore hypertools -- crashed on Windows. Fixed upstream in dw 0.5.1
# (ContextLab/data-wrangler#32, now uses ``os.path.expanduser``); we require
# >=0.5.1, but keep this zero-risk guard for environments stuck on 0.5.0.
os.environ.setdefault("HOME", os.path.expanduser("~"))

import datawrangler as dw

from .exceptions import HypertoolsIOError
from .shared import RobustDict


def _coerce(value):
    """Turn an INI string into a typed Python value where possible.

    Uses ``ast.literal_eval`` (eval-free), so ``'3'`` -> 3, ``'False'`` ->
    False, ``'0.7'`` -> 0.7, ``'None'`` -> None; non-literal strings (model
    names like ``'IncrementalPCA'``, format strings like ``'-'``) are
    returned unchanged. (2026-07 audit F23-002: raw strings like ``'3'``
    were unusable as kwargs, and ``bool('False')`` is True.)
    """
    if not isinstance(value, str):
        return value
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError, TypeError, MemoryError):
        return value


def _merge_sections(base, extra):
    """Deep-merge ``extra``'s sections into ``base`` per-option (a section in
    ``extra`` overrides only the options it names, instead of wholesale-
    replacing the section -- 2026-07 audit F23-003)."""
    for section, options in extra.items():
        base.setdefault(section, {}).update(options)
    return base


def get_default_options(fname=None):
    """Return the published defaults as a RobustDict of
    ``{section: {option: value}}``.

    The result always contains datawrangler's defaults with hypertools'
    bundled ``core/config.ini`` deep-merged on top. If ``fname`` is given,
    that file's sections are deep-merged on top of BOTH (per-option, so a
    custom ``[cluster]`` overriding one key keeps the section's other
    defaults -- it no longer silently replaces the shipped configuration).

    Parameters
    ----------
    fname : str or None
        Optional path to an additional config.ini-style file to layer on
        top of the shipped defaults. Raises
        :class:`~hypertools.core.exceptions.HypertoolsIOError` if the file
        does not exist (missing paths used to be silently ignored).

    Returns
    -------
    RobustDict
        Typed options (see ``_coerce``); unknown sections yield ``{}``.
    """
    bundled = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "config.ini")
    if not os.path.isfile(bundled):
        # a silent {} here would make every apply_defaults() lookup quietly
        # empty (2026-07 audit X5-packaging-002: config.ini was missing from
        # built wheels/sdists) -- fail loudly instead
        raise HypertoolsIOError(
            f"hypertools' bundled config file is missing: {bundled!r}. "
            "This indicates a broken installation (the package was built "
            "without its config.ini); reinstall hypertools.")

    merged = _merge_sections(dict_of_dicts(dw.core.get_default_options()),
                             dict_of_dicts(dw.core.get_default_options(bundled)))
    if fname is not None:
        if not os.path.isfile(fname):
            raise HypertoolsIOError(
                f"config file not found: {fname!r}. Pass the path to an "
                "existing config.ini-style file, or omit fname to use the "
                "defaults.")
        merged = _merge_sections(merged, dict_of_dicts(
            dw.core.get_default_options(fname)))

    # configparser always emits an (empty) 'DEFAULT' section -- an artifact,
    # not a real options section
    merged.pop("DEFAULT", None)

    for section, options in merged.items():
        merged[section] = {k: _coerce(v) for k, v in options.items()}
    return RobustDict(merged, __default_value__={})


def dict_of_dicts(options):
    """Copy a parsed config into plain nested dicts (so merging never
    mutates datawrangler's cached/parsed structures)."""
    return {section: dict(values) for section, values in options.items()}


def apply_defaults(func_name, kwargs=None):
    """Return the published defaults for ``func_name`` overridden by
    ``kwargs`` (the caller's kwargs always win; unknown ``func_name``
    yields just ``kwargs``)."""
    defaults = dict(get_default_options()[func_name])
    if kwargs:
        defaults.update(kwargs)
    return defaults
